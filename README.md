# Interpretable AI for Medical Imaging Data

The objective of this project is to provide interpretable machine and deep learning methods for medical imaging data. In safety-critical applications, professionals often exhibit reluctance to rely on neural networks when these networks lack readily interpretable explanations. This hesitation stems from the need for a clear understanding of the network's decision-making processes to ensure safety and accountability.

This work aims to bridge this gap by emphasizing the importance of human interpretability. It strives to enable experts, such as medical practitioners, to actively participate in the decision-making process. By facilitating visual interpretations of the learned concepts within the network, this approach not only enhances transparency but also ensures that domain experts can provide valuable insights and oversight, further bolstering the safety and reliability of critical systems.

In this study I am using the [NIH chest Xray](https://nihcc.app.box.com/v/ChestXray-NIHCC/) dataset but the model is applicable to any other images and applications aiming to provide interpretable predictions. I am currently working to improve a novel VAE architecture called CLAP (**C**oncept **L**earning **A**nd **P**rediction) which aims to use visually interpretable concepts as predictor for a simple classifier. 
