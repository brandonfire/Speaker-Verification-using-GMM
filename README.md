# Speaker-Verification-System
## Speaker Identification using a Gaussian Mixture model on VoxForge and SITW database.

	File Descriptions:
*train_models.py*:  
* Reading audio data of a speaker (5 utterances of speaker: belmontguy) 	
* Extracting MFCC features 
* Training GMM model
* Dumping GMM model onto Local Storage(.gmm)

*train_GMM_x.py* (x=2:5):  
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
* Read an audio(.wav) file. 
* Extract MFCC and MFCC_Deltas.
* Horizontally stacked to form a 40 dimensional feature vector.

*audio_splitting.py*:
* Read the audio file(.flac) from SITW database
* Compute duration of all audio files
* Return all audio files with duration > 1000s
* Split the audio file (75% and 25%) for training and testing
* Write the split audio (.flac) files in the

*sitw_train*:
*Load the training audio files from *audio_splitting.py*
*Extract MFCC , train Model , dump to storage

