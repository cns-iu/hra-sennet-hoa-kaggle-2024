# Code for "Vasculature segmentation in 3D hierarchical phase-contrast tomography images of human kidneys"

Yashvardhan Jain<sup>1*+</sup>, Claire L. Walsh<sup>2*+</sup>, Ekin Yagis<sup>2</sup>, Shahab Aslani<sup>2</sup>, Sonal Nandanwar<sup>2</sup>, Yang Zhou<sup>2</sup>, Juhyung Ha<sup>1</sup>, Katherine S. Gustilo<sup>1</sup>, Joseph Brunet<sup>2,3</sup>, Shahrokh Rahmani<sup>2,4</sup>, Paul Tafforeau<sup>3</sup>, Alexandre Bellier<sup>5</sup>, Griffin Weber<sup>6</sup>, Peter D. Lee<sup>2</sup>, Katy Börner<sup>1*</sup>

<sup>1</sup> Department of Intelligent Systems Engineering, Luddy School of Informatics, Computing, and Engineering, Indiana University, Bloomington, IN 47408, USA

<sup>2</sup> Department of Mechanical Engineering, University College London, London, UK

<sup>3</sup> European Synchrotron Radiation Facility, Grenoble, France

<sup>4</sup> National Heart and Lung Institute, Faculty of Medicine, Imperial College London, London, UK

<sup>5</sup> Univ. Grenoble Alpes, Department of Anatomy (LADAF), Grenoble, France

<sup>6</sup> Department of Biomedical Informatics, Harvard Medical School, Boston, MA, United States

<sup>+</sup>These authors contributed equally

<sup>*</sup>Corresponding authors

Yashvardhan Jain (yashjain@iu.edu) 

Claire Walsh (c.walsh.11@ucl.ac.uk)

Katy Börner (katy@iu.edu)

## Abstract
Efficient algorithms are needed to segment vasculature in new 3D medical imaging datasets at scale for research and clinical applications. Manual segmentation of vessels in images is time-consuming and expensive whereas computational approaches have limited accuracy. We organize a global machine learning competition, engaging 1,401 participants, to promote development of deep learning methods for 3D blood vessel segmentation in Hierarchical Phase-Contrast Tomography (HiP-CT) datasets. This paper presents a meta-analysis of the top-performing solutions, focusing on segmentation accuracy and morphological analysis. The competition and subsequent analysis reveal convergent methodological innovations: pseudo-labeling approaches that exploit data distributions, metrics and loss functions that optimize for vessel surface and topology, and multi-scale approaches that handle data heterogeneity. Additionally, the paper presents techniques for building deep learning models for the defined task, metrics to assess and compare algorithm performance, and a dataset with manually annotated and curated gold standard segmentations for future studies in blood vessel segmentation within HiP-CT imaging.

### Link to competition website: https://www.kaggle.com/competitions/blood-vessel-segmentation 

Link to Skeleton Analysis files: https://github.com/HiPCTProject/Kaggle_skeleton_analyses 
