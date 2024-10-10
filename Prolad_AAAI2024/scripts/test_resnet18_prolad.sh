################### Test Model with ProLAD ###################


# To test with URL model(MDL setting), you can uncomment below and comment the SDL script.
# python test_prolad.py --model.name=url --model.dir ./saved_results/url  --test.prolad-opt alpha+beta --model.pretrained --source ./saved_results/url --test.mode mdl --data.test omniglot vgg_flower traffic_sign mnist fungi quickdraw cu_birds cifar10 mscoco dtd cifar100 aircraft ilsvrc_2012 --out.method url --test.size 600

# test on SDL
python test_prolad.py --model.name=imagenet-net --model.dir ./saved_results/sdl --model.pretrained --source ./saved_results/sdl --test.prolad-opt alpha+beta --test.mode sdl --test.type standard --data.test omniglot vgg_flower traffic_sign mnist fungi quickdraw cu_birds cifar10 mscoco dtd cifar100 aircraft ilsvrc_2012 --out.method refined_version --test.size 600