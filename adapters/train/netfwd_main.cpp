// SPDX-License-Identifier: MIT
// Thin CLI over the netfwd library (netfwd.h): reads inputs.bin = [x (S*NIN); weights],
// runs one forward, writes outputs.bin = [global (NOUT); local (NOUT)].  Kept for parity
// testing (check_netfwd.py); the deploy path links the library directly.
//   netfwd.exe <gemm> <lin> <ln> <attn> <ew> <S> <NIN> <D> <H> <F> <NOUT> <STACK>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "netfwd.h"

int main(int argc, char **argv) {
	if (argc < 13) {
		fprintf(stderr, "usage: netfwd <gemm><lin><ln><attn><ew> S NIN D H F NOUT STACK\n");
		return 2;
	}
	NetfwdCfg c;
	c.S = atoi(argv[6]);
	c.NIN = atoi(argv[7]);
	c.D = atoi(argv[8]);
	c.H = atoi(argv[9]);
	c.F = atoi(argv[10]);
	c.NOUT = atoi(argv[11]);
	c.STACK = atoi(argv[12]);

	size_t nw = netfwd_weight_count(&c);
	std::vector<float> in((size_t)c.S * c.NIN + nw);
	FILE *f = fopen("inputs.bin", "rb");
	if (!f || fread(in.data(), sizeof(float), in.size(), f) != in.size()) {
		fprintf(stderr, "netfwd: cannot read inputs.bin\n");
		return 1;
	}
	fclose(f);

	const char *spv[5] = {argv[1], argv[2], argv[3], argv[4], argv[5]};
	NetfwdCtx *ctx = netfwd_create(&c, spv, in.data() + (size_t)c.S * c.NIN, nw);
	if (!ctx) {
		fprintf(stderr, "netfwd: create failed\n");
		return 1;
	}
	std::vector<float> out((size_t)2 * c.NOUT);
	if (!netfwd_forward(ctx, in.data(), out.data())) {
		fprintf(stderr, "netfwd: forward failed\n");
		return 1;
	}
	netfwd_destroy(ctx);

	FILE *o = fopen("outputs.bin", "wb");
	fwrite(out.data(), sizeof(float), out.size(), o);
	fclose(o);
	printf("netfwd done\n");
	return 0;
}
