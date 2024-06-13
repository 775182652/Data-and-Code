# AMultimodal Metaphorical Dataset for Emotion Classification and a Novel Method by Chain of Thought Prompt

We release the DepthSenseMET Dataset, a multimodal metaphor detection dataset consisting of 10,437 pairs of text and images, with annotations indicating the presence of metaphors. In addition we propose a  novel methodology named Metaphorical Sense Knowledge Augmentation (MSKA) which harnesses metaphorical sense features derived from the metaphor sense lexicon, thereby enhancing metaphor comprehension to unprecedented levels.
Our dataset is presented in the Ads.xlsx file, Fb.xlsx file and Twi xlsx file located in the data folder.
* The data in Ads.xlsx is derived from advertisements.
* The data in Fb.xlsx is derived from Facebook.
* The data in Twi.xlsx is derived from Twitter.

## Example Instance

.<div align='center'><img src='Ansample_cn.png' width="600" height="300"></div>

In this example, words are arranged to form a hand giving a thumbs-up, expressing appreciation for civilized behavior. Therefore, the target domain is words, the source domain is the hand, and the sentiment is trust.

## Annotation Guideline


* Our annotators identify incongruent units by examining the text and visual elements, often indicating the presence of metaphors. 
* They identify these incongruent units and interpret the irreversible "A is B" identity relationship to recognize metaphorical text-image pairs.
* By determining the primary theme that a sample intends to convey, one selects one of the two entities as the target domain, with the other entity describing the target domain referred to as the source domain.
* They identify the most appropriate emotion.




## Data Format


| Key                     |                                    Value                                    |
|-------------------------|:---------------------------------------------------------------------------:|
| `Image_id`            |                The address of the picture to be identified                  |
| `Text in Pic`     |                                          The  text in the picture                              |
| `Metaphor`|           Metaphorical judgment (0: Literal ; 1: Metaphor)          |
| `Sense`            |                       The primary categories of metaphorical sense(1: physical_property; 2: magnitude; 3: sensation; 4: shape; 5: state; 6: quality; 7: symbol )                 |


##Metaphorical Sense Lexion

*Metaphorical Sense Lexicon.dox file stores all the primary categories, the corresponding secondary categories for each primary category, and examples for each secondary category of metaphorical sense.
*The document contains a total of 7 primary categories, 129 secondary categories, and 3,724 examples.

## Run the code
In the data folder named code:
* The execution of the data_utils.py file is responsible for reading the data.
* model.py file represents the entire architecture of the model.
* Running main.py completes the training and testing process.










# Data-and-Code
The images with numbers from 9981.jpg to 11377.jpg are from the IFlytek Advertising Image Classification Competition, released in 2021, without corresponding URL are provided in file naming dataF.

Please run LLAVAcaption.py and LLAVACot_1_cn.py before code.py.
