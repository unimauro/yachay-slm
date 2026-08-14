//! Yachay SLM — inferencia en Rust puro. Sin Python, sin MLX, sin GPU.
//!
//!   yachay --prompt "¿por qué el cielo es azul?"
//!   yachay --chat
//!
//! Busca por defecto el modelo demo del repo (models/demo/yachay-demo.safetensors).

mod engine;

use std::io::{self, Write};

use engine::{load_config, Model};
use tokenizers::Tokenizer;

const DEMO: &str = "models/demo/yachay-demo.safetensors";

/// RNG determinista (xorshift64*) para --seed reproducible.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let v = x.wrapping_mul(0x2545F4914F6CDD1D) >> 40; // 24 bits
        (v as f32) / ((1u32 << 24) as f32)
    }
}

fn softmax(logits: &mut [f32]) {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for l in logits.iter_mut() {
        *l = (*l - m).exp();
        sum += *l;
    }
    for l in logits.iter_mut() {
        *l /= sum;
    }
}

fn sample(mut logits: Vec<f32>, temp: f32, top_k: usize, rng: &mut Rng) -> u32 {
    let t = temp.max(1e-6);
    for l in logits.iter_mut() {
        *l /= t;
    }
    if top_k > 0 && top_k < logits.len() {
        let mut sorted = logits.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let kth = sorted[top_k - 1];
        for l in logits.iter_mut() {
            if *l < kth {
                *l = f32::NEG_INFINITY;
            }
        }
    }
    softmax(&mut logits);
    let r = rng.next_f32();
    let mut acc = 0.0f32;
    for (i, &p) in logits.iter().enumerate() {
        acc += p;
        if r < acc {
            return i as u32;
        }
    }
    (logits.len() - 1) as u32
}

fn generate(
    model: &Model,
    tok: &Tokenizer,
    prompt: &str,
    max_new: usize,
    temp: f32,
    top_k: usize,
    rng: &mut Rng,
) -> String {
    let eos = tok.token_to_id("<eos>");
    let start = tok.encode(format!("<bos>{}\n", prompt), false).unwrap();
    let mut ids: Vec<u32> = start.get_ids().to_vec();
    let block = model.cfg.block_size;
    for _ in 0..max_new {
        let ctx: Vec<u32> = if ids.len() > block {
            ids[ids.len() - block..].to_vec()
        } else {
            ids.clone()
        };
        let logits = model.forward_last(&ctx);
        let nxt = sample(logits, temp, top_k, rng);
        ids.push(nxt);
        if Some(nxt) == eos {
            break;
        }
    }
    tok.decode(&ids, true).unwrap()
}

fn seed_default() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
    };
    let has = |flag: &str| args.iter().any(|a| a == flag);

    let mut ckpt = get("--ckpt").unwrap_or_else(|| DEMO.to_string());
    if !std::path::Path::new(&ckpt).exists() && std::path::Path::new(DEMO).exists() {
        eprintln!("[no encontré {}; uso el demo: {}]", ckpt, DEMO);
        ckpt = DEMO.to_string();
    }
    let max_new: usize = get("--max_new").and_then(|s| s.parse().ok()).unwrap_or(120);
    let temp: f32 = get("--temp").and_then(|s| s.parse().ok()).unwrap_or(0.8);
    let top_k: usize = get("--top_k").and_then(|s| s.parse().ok()).unwrap_or(40);
    let seed: u64 = get("--seed").and_then(|s| s.parse().ok()).unwrap_or_else(seed_default);

    let (cfg, tok_path) = load_config(&ckpt).expect("no pude leer el config .json");
    let model = Model::load(&ckpt, cfg).expect("no pude cargar los pesos");
    let tok = Tokenizer::from_file(&tok_path).expect("no pude cargar el tokenizer");

    // Modo oculto para el test de paridad entre lenguajes.
    if let Some(csv) = get("--dump-logits") {
        let ids: Vec<u32> = csv.split(',').map(|s| s.trim().parse().unwrap()).collect();
        let logits = model.forward_last(&ids);
        let out: Vec<String> = logits.iter().map(|v| format!("{:.6}", v)).collect();
        println!("{}", out.join(" "));
        return;
    }

    let mut rng = Rng::new(seed);
    eprintln!(
        "[modelo Rust cargado: {} capas, dim {}, vocab {} | sin Python]",
        model.cfg.n_layers, model.cfg.dim, model.cfg.vocab_size
    );

    if has("--chat") {
        println!("Modo chat. Escribe 'salir' para terminar.");
        loop {
            print!("\n> ");
            io::stdout().flush().unwrap();
            let mut line = String::new();
            if io::stdin().read_line(&mut line).unwrap() == 0 {
                break;
            }
            let p = line.trim();
            if p.eq_ignore_ascii_case("salir") || p.eq_ignore_ascii_case("exit") || p.eq_ignore_ascii_case("quit") {
                break;
            }
            if !p.is_empty() {
                println!("{}", generate(&model, &tok, p, max_new, temp, top_k, &mut rng));
            }
        }
    } else if let Some(prompt) = get("--prompt") {
        println!("{}", generate(&model, &tok, &prompt, max_new, temp, top_k, &mut rng));
    } else {
        eprintln!("Da un --prompt \"...\" o usa --chat");
        std::process::exit(1);
    }
}
