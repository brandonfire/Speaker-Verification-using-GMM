# Speaker-Identification-using-GMM
## Speaker Identification using a Gaussian Mixture model on VoxForge dataset.

	File Descriptions:
*train_models.py*:  
* Reading audio data of a speaker (5 utterances of speaker: belmontguy) 	
* Extracting MFCC features 
* Training GMM model
* Dumping GMM model onto Local Storage(.gmm)

*train_GMM_x.py*(x=2:5):  
* Reading audio data of a speaker (5 utterances of speaker) 	
* Extracting MFCC features 
* Training GMM model
* Dumping GMM model onto Local Storage(.gmm)

*test.py*:
* Read audio files of 5 test speakers using Scipy
* Read the GMM model(.gmm) from local storage  
* Test against all speakers 
* Computer Winner speaker based on log_likelihood

*delta_40_extraction.py*:
* Read an audio(.wav) file 
* Extract MFCC and Deltas
* Combine the two to form a 40dimensional feature vector for Audio Analysis
