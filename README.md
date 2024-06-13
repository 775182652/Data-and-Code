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
| `Img_id`            |                The address of the image to be identified                  |
| `Img_url`            |                The URL for downloading the image                  |
| `Text`     |                                          The  text in the image                              |
| `Metaphor`|           Metaphorical judgment (0: Literal ; 1: Metaphor)          |
| `Target domain`|           Target domain in the metaphorical sample         |
| `Source domain`|           Source domain in the metaphorical sample          |
| `Emotion`            |                      The emotion of the metaphorical sample(1joy ; 2love ; 3trust ; 4fear ; 5sadness ; 6disgust ; 7anger ; 8surprise ; 9anticipation ; 10neutral)                 |

* The images with numbers from 9981.jpg to 11377.jpg are from the IFlytek Advertising Image Classification Competition, released in 2021, without corresponding URL are provided in file naming dataF.

## Run the code
* LLAVAcaption.py is responsible for extracting captions from images.
* LLAVACot_1_cn.py is responsible for getting Chain of thought prompt result.
* code.py is the completes the training and testing process









