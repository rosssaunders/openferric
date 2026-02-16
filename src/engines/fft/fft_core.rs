use std::f64::consts::PI;

use num_complex::Complex;

#[inline]
fn bit_reverse(mut x: usize, bits: u32) -> usize {
    let mut y = 0_usize;
    for _ in 0..bits {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    y
}

fn fft_inplace(values: &mut [Complex<f64>], inverse: bool) {
    let n = values.len();
    assert!(n.is_power_of_two(), "fft length must be power-of-two");

    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = bit_reverse(i, bits);
        if j > i {
            values.swap(i, j);
        }
    }

    let mut len = 2;
    while len <= n {
        let angle = if inverse {
            2.0 * PI / len as f64
        } else {
            -2.0 * PI / len as f64
        };
        let wlen = Complex::new(0.0, angle).exp();

        let half = len / 2;
        let mut i = 0;
        while i < n {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..half {
                let u = values[i + j];
                let v = values[i + j + half] * w;
                values[i + j] = u + v;
                values[i + j + half] = u - v;
                w *= wlen;
            }
            i += len;
        }

        len <<= 1;
    }

    if inverse {
        let scale = n as f64;
        for x in values.iter_mut() {
            *x /= scale;
        }
    }
}

pub fn fft_forward(values: &mut [Complex<f64>]) {
    fft_inplace(values, false);
}

pub fn fft_inverse(values: &mut [Complex<f64>]) {
    fft_inplace(values, true);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_inverse_roundtrip() {
        let original = vec![
            Complex::new(1.0, 0.0),
            Complex::new(-2.0, 0.5),
            Complex::new(3.0, -1.5),
            Complex::new(0.2, 0.1),
            Complex::new(-0.7, 0.0),
            Complex::new(0.3, -0.9),
            Complex::new(0.0, 0.0),
            Complex::new(2.5, 1.1),
        ];

        let mut transformed = original.clone();
        fft_forward(&mut transformed);
        fft_inverse(&mut transformed);

        for (lhs, rhs) in transformed.iter().zip(original.iter()) {
            assert!((*lhs - *rhs).norm() < 1e-10);
        }
    }
}
