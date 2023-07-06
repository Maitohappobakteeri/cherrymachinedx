NUM_STEPS="$(ls cache/v1/0/ | wc -l)"

# rm -rf .montage-tmp
[[ -d .montage-tmp ]] || mkdir .montage-tmp

I=0
for i in $(seq -f "%012g" 0 $(( $NUM_STEPS - 1 )))
do
    echo -en "\r$I/$NUM_STEPS"
    I=$((I+1))
    if [ -f ".montage-tmp/$i.png" ]; then
        continue
    fi
    montage cache/v1/*/$i.png -geometry +0x0 ".montage-tmp/$i.png"
done
echo ""
ffmpeg -y -framerate 30 -pattern_type glob -i '.montage-tmp/*.png' -c:v libx264 -vf "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1" -pix_fmt yuv420p montage.mp4