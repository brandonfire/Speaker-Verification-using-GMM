# Speaker-Identification-using-GMM
Speaker Verification using GMM_UBM and i-vectors using VoxForge dataset
Code Valid for Linux Machines only. For Windows users, change of path and slashes are required in several places.

	File Descriptions:
	train_models.py:  
1) Reading audio data of 34 speakers (5 utterances per speaker) in Python	
2) Extracting MFCC Features 
3) Training GMM model
4) Dumping GMM model onto Local Storage(.gmm)

       test.py:
1) Read GMM Files
2) Test 
3) Compute winner speaker

       test_delta.py
1) Used for calculating the delta co-eff from MFCC using python_speech_features package.

	speakerfeatures.py
1) Used for computing combined mfcc and delta (40 dimensional vector)	
			
	
