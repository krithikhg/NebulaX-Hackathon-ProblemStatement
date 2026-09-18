// Complex FFT for arbitrary lengths: mixed radix for small prime factors,
// Bluestein's algorithm when a large prime factor would make that slow.

function factorize(n) {
  const f = [];
  for (const p of [4, 2, 3, 5]) {
    while (n % p === 0) { f.push(p); n /= p; }
  }
  for (let p = 7; p * p <= n; p += 2) {
    while (n % p === 0) { f.push(p); n /= p; }
  }
  if (n > 1) f.push(n);
  return f;
}

// Recursive decimation-in-time over the factor list.
function mixedRadix(re, im, n, factors) {
  const outRe = new Float64Array(n);
  const outIm = new Float64Array(n);
  const twRe = new Float64Array(n);
  const twIm = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const a = (-2 * Math.PI * k) / n;
    twRe[k] = Math.cos(a);
    twIm[k] = Math.sin(a);
  }

  function rec(inOff, stride, len, fi, oOff) {
    if (len === 1) {
      outRe[oOff] = re[inOff];
      outIm[oOff] = im[inOff];
      return;
    }
    const p = factors[fi];
    const m = len / p;
    for (let q = 0; q < p; q++) rec(inOff + q * stride, stride * p, m, fi + 1, oOff + q * m);
    // Butterfly: combine p sub-transforms of length m.
    const tRe = new Float64Array(p);
    const tIm = new Float64Array(p);
    const step = n / len;
    for (let k = 0; k < m; k++) {
      for (let q = 0; q < p; q++) {
        const idx = oOff + q * m + k;
        const w = (q * k * step) % n;
        const xr = outRe[idx];
        const xi = outIm[idx];
        tRe[q] = xr * twRe[w] - xi * twIm[w];
        tIm[q] = xr * twIm[w] + xi * twRe[w];
      }
      for (let s = 0; s < p; s++) {
        let sr = 0;
        let si = 0;
        for (let q = 0; q < p; q++) {
          const w = (((q * s * m) % len) * step) % n;
          sr += tRe[q] * twRe[w] - tIm[q] * twIm[w];
          si += tRe[q] * twIm[w] + tIm[q] * twRe[w];
        }
        outRe[oOff + s * m + k] = sr;
        outIm[oOff + s * m + k] = si;
      }
    }
  }

  rec(0, 1, n, 0, 0);
  return [outRe, outIm];
}

function bluestein(re, im, n) {
  let m = 1;
  while (m < 2 * n - 1) m *= 2;
  const cRe = new Float64Array(n);
  const cIm = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const a = (Math.PI * ((k * k) % (2 * n))) / n;
    cRe[k] = Math.cos(a);
    cIm[k] = -Math.sin(a);
  }
  const aRe = new Float64Array(m);
  const aIm = new Float64Array(m);
  const bRe = new Float64Array(m);
  const bIm = new Float64Array(m);
  for (let k = 0; k < n; k++) {
    aRe[k] = re[k] * cRe[k] - im[k] * cIm[k];
    aIm[k] = re[k] * cIm[k] + im[k] * cRe[k];
  }
  bRe[0] = cRe[0];
  bIm[0] = -cIm[0];
  for (let k = 1; k < n; k++) {
    bRe[k] = bRe[m - k] = cRe[k];
    bIm[k] = bIm[m - k] = -cIm[k];
  }
  const f = factorize(m);
  const [ARe, AIm] = mixedRadix(aRe, aIm, m, f);
  const [BRe, BIm] = mixedRadix(bRe, bIm, m, f);
  const pRe = new Float64Array(m);
  const pIm = new Float64Array(m);
  for (let k = 0; k < m; k++) {
    // conjugate the product so a forward FFT computes the inverse
    pRe[k] = ARe[k] * BRe[k] - AIm[k] * BIm[k];
    pIm[k] = -(ARe[k] * BIm[k] + AIm[k] * BRe[k]);
  }
  const [qRe, qIm] = mixedRadix(pRe, pIm, m, f);
  const outRe = new Float64Array(n);
  const outIm = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const xr = qRe[k] / m;
    const xi = -qIm[k] / m;
    outRe[k] = xr * cRe[k] - xi * cIm[k];
    outIm[k] = xr * cIm[k] + xi * cRe[k];
  }
  return [outRe, outIm];
}

/** |rfft(x)|^2 for bins 0..floor(n/2), like numpy.abs(numpy.fft.rfft(x))**2. */
export function rfftPower(x) {
  const n = x.length;
  const re = Float64Array.from(x);
  const im = new Float64Array(n);
  const factors = factorize(n);
  const [oRe, oIm] = factors[factors.length - 1] > 64 ? bluestein(re, im, n) : mixedRadix(re, im, n, factors);
  const half = Math.floor(n / 2) + 1;
  const p = new Float64Array(half);
  for (let k = 0; k < half; k++) p[k] = oRe[k] * oRe[k] + oIm[k] * oIm[k];
  return p;
}
