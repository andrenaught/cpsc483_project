#Importing the essential libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#1. Import dataset
cols_to_use = [1,2,3,4,5,6,7,8,9,10,11,12,13,18,19,21] #we'll ignore some columns
dataset = pd.read_csv("drug_consumption.data.txt", sep=",", header=None, usecols=cols_to_use)
dataset.columns = ["Age", "Gender", "Education", "Country", "Ethnicity", #1,2,3,4,5
				   "Nscore", "Escore", "Oscore", "Ascore", "Cscore", "Impulsive", "Sensation_seeking", #6,7,8,9,10,11,12
				   "Alcohol_cons", "Cannabis_cons", "Chocolate_cons","Crack_cons"] #13, 18, 19, 21
print (dataset)

'''
***** PARTICIPANT DATA *****
1. Age - their age (ex: 0.49788)
2. Gender - their gender (ex: 0.48246)
3. Education - level of education (ex: -0.05921)
4. Country - country of residence (ex: 0.96082)
5. Ethnicity - their ethnicity (ex: 0.12600)

***** PERSONALITY TRAIT DATA *****
6. Nscore - Personality trait 'Neuroticism' score (ex: 0.31287)
	People who are neurotic respond worse to stressors and are more likely to interpret 
	ordinary situations as threatening and minor frustrations as hopelessly difficult.

7. Escore - Personality trait 'Extraversion' score (ex: -0.57545)
	Energetic, surgency, assertiveness, sociability and the tendency 
	to seek stimulation in the company of others, and talkativeness.

8. Oscore - Personality trait 'Openness to experience' score (ex: -0.58331)
	Appreciation for art, emotion, adventure, unusual ideas, curiosity, and variety of experience.

9. Ascore - Personality trait 'Agreeableness' score (ex: -0.91699)
	Tendency to be compassionate and cooperative rather than suspicious and antagonistic towards others.

10. Cscore - Personality trait 'Conscientiousness' score (ex: -0.00665)
	Tendency to be organized and dependable, show self-discipline, act dutifully, 
	aim for achievement, and prefer planned rather than spontaneous behavior.

11. Impulsive - Personality trait 'Impulsiveness' score (ex: -0.21712)

12. Sensation_seeking - Personality trait 'Senation Seeking' score (ex: -1.18084)

***** DRUG CONSUMPTION *****
13. Alcohol_cons - alcohol consumption (ex: CL5)
	...
18. Cannabis_cons - cannabis consumption (ex: CL0)
19. Choc_cons - chocolate consumption (ex: CL5)
	...
21. Crack_cons - crack consumption (CL0)
'''

#It looks like 0 - 12 is already normalized and feature encoded - so don't have to worry about that.
#TODO: feature encode + normalize 13, 18, 19, 21
#TODO: check for missing values - replace with average
