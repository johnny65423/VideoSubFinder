#!/usr/bin/env bash
# T5: end-to-end comparison of the CPU and the GPU transform path with the real program.
#
#   e2e_compare.sh <VideoSubFinderWXW executable> <video> <start> <end> [work dir]
#
# start / end are in the format of the -s / -e options (h:mm:ss:mmm). The search (-c -r) runs twice, with
# -use_cuda_gpu_transform=0 and =1, on a copy of the general settings of the executable. Then everything the search
# wrote (RGBImages, ISAImages, ILAImages ...) is compared byte for byte.
# Exit code 0: identical, 1: different, 2: a run failed.
#
# Git Bash on Windows rewrites arguments that look like paths (0:01:00:000, /c ...): the conversion is switched off below.
set -u
export MSYS2_ARG_CONV_EXCL="*"

exe="${1:?executable}"; video="${2:?video}"; start="${3:?start}"; end="${4:?end}"
work="${5:-./gpu_parity_e2e}"
# the program is a native executable: it needs Windows paths
winpath() { cygpath -m "$1" 2>/dev/null || echo "$1"; }
video="$(winpath "$video")"
exe_dir="$(cd "$(dirname "$exe")" && pwd)"
mkdir -p "$work"; work="$(cd "$work" && { pwd -W 2>/dev/null || pwd; })"   # a Windows path: the program is a native executable

cp "$exe_dir/settings/general.cfg" "$work/general.cfg" 2>/dev/null || { echo "no settings/general.cfg next to the executable"; exit 2; }

run() {   # mode
	local mode="$1" out="$work/out$1"
	rm -rf "$out"; mkdir -p "$out"
	local t0 t1
	t0=$(date +%s%N)
	# the program ends with exit code 255 also when everything went well (OnInit returns false), so the code is not checked
	( cd "$exe_dir" && "./$(basename "$exe")" -c -r -i "$video" -o "$out" -gs "$work/general.cfg" -s "$start" -e "$end" -use_cuda_gpu_transform="$mode" >"$work/log$mode.txt" 2>&1 )
	t1=$(date +%s%N)
	local files
	files=$(find "$out/RGBImages" -type f 2>/dev/null | wc -l)
	awk -v m="$mode" -v a="$t0" -v b="$t1" -v f="$files" 'BEGIN { printf "search  use_cuda_gpu_transform=%s : %.1f s, %d subtitle images\n", m, (b - a) / 1e9, f }'
	[ "$files" -gt 0 ] || { echo "no images were written (see $work/log$mode.txt)"; exit 2; }
}

run 0
run 1

if diff -rq "$work/out0" "$work/out1" >"$work/diff.txt"; then
	echo "IDENTICAL: $(find "$work/out0" -type f | wc -l) files"
	exit 0
fi
echo "DIFFERENT:"; head -20 "$work/diff.txt"
exit 1
