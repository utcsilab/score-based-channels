python simplified_inference.py --gpu 0 --save_channels 1 --channel CDL-C --noise 0.001 --spacing 0.5 --pilot_alpha 1.0 0.8 0.6 &
python simplified_inference.py --gpu 1 --save_channels 1 --channel CDL-B --noise 0.001 --spacing 0.5 --pilot_alpha 1.0 0.8 0.6 &
python simplified_inference.py --gpu 2 --save_channels 1 --channel CDL-D --noise 0.001 --spacing 0.5 --pilot_alpha 1.0 0.8 0.6 &