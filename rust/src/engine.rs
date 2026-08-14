//! Motor de inferencia en Rust puro. Reproduce exactamente `src/portable/engine.py`.
//! Carga pesos .safetensors (entrenados con MLX) y corre en cualquier CPU.

use std::collections::HashMap;
use std::path::Path;

use safetensors::SafeTensors;

/// Config del modelo (viene del sidecar .json que guarda el entrenamiento).
pub struct Config {
    pub vocab_size: usize,
    pub block_size: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
}

/// Pesos: nombre -> vector plano f32 (row-major, igual que safetensors/PyTorch).
pub struct Model {
    w: HashMap<String, Vec<f32>>,
    pub cfg: Config,
    head_dim: usize,
}

fn erf(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26 (mismo que engine.py, error < 1.5e-7).
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

impl Model {
    pub fn load(ckpt: &str, cfg: Config) -> Result<Model, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(ckpt)?;
        let st = SafeTensors::deserialize(&bytes)?;
        let mut w = HashMap::new();
        for name in st.names() {
            let t = st.tensor(name)?;
            let data = t.data();
            let mut v = Vec::with_capacity(data.len() / 4);
            for chunk in data.chunks_exact(4) {
                v.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            w.insert(name.to_string(), v);
        }
        let head_dim = cfg.dim / cfg.n_heads;
        Ok(Model { w, cfg, head_dim })
    }

    fn g(&self, k: &str) -> &[f32] {
        self.w.get(k).unwrap_or_else(|| panic!("falta peso: {}", k))
    }

    /// y = x @ W^T + b   con x:(t,in), W:(out,in) row-major -> y:(t,out)
    fn linear(&self, x: &[f32], t: usize, in_d: usize, w: &[f32], b: Option<&[f32]>, out_d: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; t * out_d];
        for i in 0..t {
            let xr = &x[i * in_d..i * in_d + in_d];
            for o in 0..out_d {
                let wr = &w[o * in_d..o * in_d + in_d];
                let mut acc = 0.0f32;
                for k in 0..in_d {
                    acc += xr[k] * wr[k];
                }
                if let Some(bb) = b {
                    acc += bb[o];
                }
                y[i * out_d + o] = acc;
            }
        }
        y
    }

    fn layernorm(&self, x: &[f32], t: usize, dim: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
        let eps = 1e-5f32;
        let mut y = vec![0.0f32; t * dim];
        for i in 0..t {
            let row = &x[i * dim..i * dim + dim];
            let mean: f32 = row.iter().sum::<f32>() / dim as f32;
            let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for d in 0..dim {
                y[i * dim + d] = (row[d] - mean) * inv * w[d] + b[d];
            }
        }
        y
    }

    fn attention(&self, x: &[f32], t: usize, li: usize) -> Vec<f32> {
        let dim = self.cfg.dim;
        let nh = self.cfg.n_heads;
        let hd = self.head_dim;
        let p = format!("blocks.{}.attn.", li);
        let q = self.linear(x, t, dim, self.g(&format!("{}query_proj.weight", p)), Some(self.g(&format!("{}query_proj.bias", p))), dim);
        let k = self.linear(x, t, dim, self.g(&format!("{}key_proj.weight", p)), Some(self.g(&format!("{}key_proj.bias", p))), dim);
        let v = self.linear(x, t, dim, self.g(&format!("{}value_proj.weight", p)), Some(self.g(&format!("{}value_proj.bias", p))), dim);

        let scale = 1.0f32 / (hd as f32).sqrt();
        let mut out = vec![0.0f32; t * dim]; // concatenación de cabezas
        for h in 0..nh {
            let off = h * hd;
            for i in 0..t {
                // scores causales sobre j=0..=i, con softmax
                let qi = &q[i * dim + off..i * dim + off + hd];
                let mut scores = vec![0.0f32; i + 1];
                let mut maxs = f32::NEG_INFINITY;
                for j in 0..=i {
                    let kj = &k[j * dim + off..j * dim + off + hd];
                    let mut acc = 0.0f32;
                    for d in 0..hd {
                        acc += qi[d] * kj[d];
                    }
                    let s = acc * scale;
                    scores[j] = s;
                    if s > maxs {
                        maxs = s;
                    }
                }
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - maxs).exp();
                    sum += *s;
                }
                // salida ponderada
                for d in 0..hd {
                    let mut acc = 0.0f32;
                    for j in 0..=i {
                        acc += (scores[j] / sum) * v[j * dim + off + d];
                    }
                    out[i * dim + off + d] = acc;
                }
            }
        }
        // proyección de salida
        self.linear(&out, t, dim, self.g(&format!("{}out_proj.weight", p)), Some(self.g(&format!("{}out_proj.bias", p))), dim)
    }

    fn block(&self, x: &mut Vec<f32>, t: usize, li: usize) {
        let dim = self.cfg.dim;
        let p = format!("blocks.{}.", li);
        // atención + residual
        let h = self.layernorm(x, t, dim, self.g(&format!("{}ln1.weight", p)), self.g(&format!("{}ln1.bias", p)));
        let a = self.attention(&h, t, li);
        for i in 0..x.len() {
            x[i] += a[i];
        }
        // mlp + residual
        let h = self.layernorm(x, t, dim, self.g(&format!("{}ln2.weight", p)), self.g(&format!("{}ln2.bias", p)));
        let hidden = 4 * dim;
        let mut m = self.linear(&h, t, dim, self.g(&format!("{}mlp.layers.0.weight", p)), Some(self.g(&format!("{}mlp.layers.0.bias", p))), hidden);
        for v in m.iter_mut() {
            *v = gelu(*v);
        }
        let m = self.linear(&m, t, hidden, self.g(&format!("{}mlp.layers.2.weight", p)), Some(self.g(&format!("{}mlp.layers.2.bias", p))), dim);
        for i in 0..x.len() {
            x[i] += m[i];
        }
    }

    /// Forward completo. Devuelve logits de la ÚLTIMA posición: Vec<f32> de vocab_size.
    pub fn forward_last(&self, ids: &[u32]) -> Vec<f32> {
        let dim = self.cfg.dim;
        let t = ids.len();
        let tok = self.g("tok_emb.weight");
        let pos = self.g("pos_emb.weight");
        let mut x = vec![0.0f32; t * dim];
        for (i, &id) in ids.iter().enumerate() {
            let tr = &tok[id as usize * dim..id as usize * dim + dim];
            let pr = &pos[i * dim..i * dim + dim];
            for d in 0..dim {
                x[i * dim + d] = tr[d] + pr[d];
            }
        }
        for li in 0..self.cfg.n_layers {
            self.block(&mut x, t, li);
        }
        let x = self.layernorm(&x, t, dim, self.g("ln_f.weight"), self.g("ln_f.bias"));
        // logits solo de la última posición
        let last = &x[(t - 1) * dim..t * dim];
        let head = self.g("head.weight");
        let vocab = self.cfg.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        for vv in 0..vocab {
            let wr = &head[vv * dim..vv * dim + dim];
            let mut acc = 0.0f32;
            for d in 0..dim {
                acc += last[d] * wr[d];
            }
            logits[vv] = acc;
        }
        logits
    }
}

/// Lee el sidecar .json (junto al .safetensors) y arma el Config + ruta del tokenizer.
pub fn load_config(ckpt: &str) -> Result<(Config, String), Box<dyn std::error::Error>> {
    let json_path = Path::new(ckpt).with_extension("json");
    let txt = std::fs::read_to_string(&json_path)?;
    let v: serde_json::Value = serde_json::from_str(&txt)?;
    let m = &v["model"];
    let cfg = Config {
        vocab_size: m["vocab_size"].as_u64().unwrap() as usize,
        block_size: m["block_size"].as_u64().unwrap() as usize,
        dim: m["dim"].as_u64().unwrap() as usize,
        n_layers: m["n_layers"].as_u64().unwrap() as usize,
        n_heads: m["n_heads"].as_u64().unwrap() as usize,
    };
    let tok = v["tokenizer"].as_str().unwrap_or("tokenizer.json").to_string();
    Ok((cfg, tok))
}
