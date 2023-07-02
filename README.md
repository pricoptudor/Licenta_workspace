<a name="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- PROJECT DETAILS -->
<br />
<div align="center">
    <h3 align="center">🎵 Enhancing our lives through music: <br> Genre Classification and Generator</h3>

  <p align="center">
    <br />
    <a href="https://github.com/pricoptudor/Licenta_workspace/blob/main/Licenta%20-%20Music.pdf"><strong>Explore the documentation »</strong></a>
    <br />
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#license">License</a></li>
  </ol>
  <br />
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<br />

This repository contains the notebooks and data used in my Bachelor's thesis project at <i><b>FII - Cuza University Iasi</b></i>. The goal of this project was two-fold:

<ul>
    <li>build a music genre classification model that could accurately categorize music tracks into their respective genres;</li>
    <li>develop a music generation methodology that could create new, pleasant music based on the learnt characteristics of a specific genre;</li>
</ul>

<br/>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<br/>


<!-- GETTING STARTED -->
## Getting Started

This project is organized into 4 main directories:

<ol>
    <li><a href="https://github.com/pricoptudor/Licenta_workspace/tree/main/Project_1">Project_1</a>: Data analysis, Machine Learning, Deep Learning on <b>GTZAN</b> dataset.</li>
    <li><a href="https://github.com/pricoptudor/Licenta_workspace/tree/main/Project_2">Project_2</a>: Data analysis on <b>FMA</b> dataset.</li>
    <li><a href="https://github.com/pricoptudor/Licenta_workspace/tree/main/Project_3">Project_3</a>: Data generation, Data analysis, Machine Learning, Deep Learning on <b>Spotify</b> dataset.</li>
    <li><a href="https://github.com/pricoptudor/Licenta_workspace/tree/main/Project_4">Project_4</a>: Music generation with <b>VAEs</b>.</li>
</ol>

This collection of notebooks was built and tuned inside an environment from Anaconda with `Python==3.10.9`. Few of the technologies used:

* [![Tensorflow][Tensorflow]][Tensorflow-url]
* [![Keras][Keras]][Keras-url]
* [![Jupyter][Jupyter]][Jupyter-url]
* [![Anaconda][Anaconda]][Anaconda-url]

<br />

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

For each project, download the dataset required for experimentation:

<ul>
    <li><a href="https://github.com/pricoptudor/Licenta_workspace/tree/main/Project_1">Project_1</a>: <b>GTZAN</b> dataset: <a href="https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification">Explore GTZAN</a> </li>
    <li><a href="https://github.com/pricoptudor/Licenta_workspace/tree/main/Project_2">Project_2</a>: <b>FMA</b> dataset: <a href="https://github.com/mdeff/fma">Explore FMA</a></li>
    <li><a href="https://github.com/pricoptudor/Licenta_workspace/tree/main/Project_3">Project_3</a>: <b>Spotify</b> dataset: <a href="https://www.kaggle.com/datasets/pricoptudor/spotify-dataset">Explore Spotify features</a> or <a href="https://www.kaggle.com/datasets/pricoptudor/spotify-image-dataset">Explore Spotify melspectrograms</a></li>
    <li><a href="https://github.com/pricoptudor/Licenta_workspace/tree/main/Project_4">Project_4</a>: <b>Jazz</b> dataset: <a href="https://www.kaggle.com/datasets/saikayala/jazz-ml-ready-midi?select=Jazz-midi.csv">Explore Jazz</a></li>
</ul>

<br/>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- RESULTS -->
## Results


VAE-generated: 
- [Bad example](https://github.com/pricoptudor/Licenta_workspace/blob/main/Project_4/VAE_bad.mp3)
- [Good with 0.7 temperature](https://github.com/pricoptudor/Licenta_workspace/blob/main/Project_4/VAE_good_0.7.mp3)

TransformerVAE-generated:
- [Good with 0.3 temperature](https://github.com/pricoptudor/Licenta_workspace/blob/main/Project_4/TransformerVAE_good_0.3.mp3)

<br/>

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Pricop Tudor-Constantin - pricoptudor2001@gmail.com

Github: [Profile page](https://github.com/pricoptudor)


<br/>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<br/>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/pricoptudor/Licenta_workspace.svg?style=for-the-badge
[contributors-url]: https://github.com/pricoptudor/Licenta_workspace/graphs/contributors
[license-shield]: https://img.shields.io/github/license/pricoptudor/Licenta_workspace.svg?style=for-the-badge
[license-url]: https://github.com/pricoptudor/Licenta_workspace/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/tudorc-pricop/
[Tensorflow]: https://img.shields.io/badge/tensorflow-000000?style=for-the-badge&logo=tensorflow&logoColor=white
[Tensorflow-url]: https://www.tensorflow.org/
[Keras]: https://img.shields.io/badge/Keras-DD0031?style=for-the-badge&logo=keras&logoColor=white
[Keras-url]: https://keras.io/
[Jupyter]: https://img.shields.io/badge/Jupyter-4A4A55?style=for-the-badge&logo=jupyter&logoColor=FF3E00
[Jupyter-url]: https://jupyter.org/
[Anaconda]: https://img.shields.io/badge/Anaconda-0769AD?style=for-the-badge&logo=anaconda&logoColor=white
[Anaconda-url]: https://anaconda.org/ 