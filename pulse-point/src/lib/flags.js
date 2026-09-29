// URL switches for diagnosing phone memory issues: ?nodepth=1 disables depth, ?cpu=1 skips WebGPU.
const params = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : new URLSearchParams();

export const FLAGS = {
  noDepth: params.has('nodepth'),
  forceCpu: params.has('cpu'),
};
