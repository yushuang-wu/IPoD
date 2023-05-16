CLASS=chair
ROOT=/data

python 1_scale.py --in_dir=ROOT/CLASS/raw \
                  --out_dir=ROOT/CLASS/1_scaled/

python 2_fusion.py --mode=render --n_views 200 --resolution 512 \
                --in_dir=ROOT/CLASS/1_scaled/ \
                --depth_dir=ROOT/CLASS/2_depth/ \
                --out_dir=ROOT/CLASS/2_watertight/

python 2_fusion.py --mode=fuse  --n_views 200 --resolution 512 \
                --in_dir=ROOT/CLASS/1_scaled/  \
                --depth_dir=ROOT/CLASS/2_depth/ \
                --out_dir=ROOT/CLASS/2_watertight/ \

python 3_simplify.py --in_dir=ROOT/CLASS/2_watertight/ \
                --out_dir=ROOT/CLASS/3_simplify/