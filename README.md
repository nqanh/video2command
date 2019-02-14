## [Translating Videos to Commands for Robotic Manipulation with Deep Recurrent Neural Networks](https://arxiv.org/pdf/1710.00290.pdf)
By A Nguyen et al. - ICRA 2018

![video2command](https://github.com/nqanh/video2command/blob/master/mics/Video2command.jpg =200x80 "video2command")

### Requirements
- Tensorflow >= 1.0 (used 1.1.0)


### Training
- Clone the repository to your `$VC_Folder`
- We train the network using [IIT-V2C dataset](https://sites.google.com/site/iitv2c/)
- You can extract the features for each frames in the input videos using any network (e.g., VGG, ResNet, etc.)
- For a quick start, the pre-extracted features with ResNet50 is available [here](#). 
- Extract the file you downloaded to `$VC_Folder/data`
- Start training: `python train.py` in `$VC_Folder/main`


### Predict & evaluate
- Predict: `python predict.py` in `$VC_Folder/main` folder
- Prepare the results for evaluation: `python prepare_evaluation_format.py` in `$VC_Folder/evaluation` folder
- Evaluate: `python cocoeval.py` in `$VC_Folder/evaluation` folder


If you find this code useful in your research, please consider citing:

	@inproceedings{nguyen2018translating,
	  title={Translating videos to commands for robotic manipulation with deep recurrent neural networks},
	  author={Nguyen, Anh and Kanoulas, Dimitrios and Muratore, Luca and Caldwell, Darwin G and Tsagarakis, Nikos G},
	  booktitle={2018 IEEE International Conference on Robotics and Automation (ICRA)},
	  year={2018},
	  organization={IEEE}
	}


### Contact
If you have any questions or comments, please send an to `anh.nguyen@iit.it`

