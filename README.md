# TCC
Trabalho de conclusão de curso, visando o cumprimento da disciplina PSG.
<br />
<br />


## Comandos Linux
Os comandos a seguir, executados no terminal do Linux, podem ser utilizados para criar o ambiente de desenvolvimento e de teste na linguagem Python.

#### Compilador C
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential
sudo gcc -v
sudo make -v
```
#### Pip3
```
sudo apt-get install python3-setuptools
sudo easy_install3 pip
```
#### Python3 dev kit
```
sudo apt-get install python3.5-dev
```
#### Cython
```
sudo pip3 install Cython
```
#### Scipy
```
sudo pip3 install scipy
```
#### Pyemd
```
sudo pip3 install pyemd
```
#### Pandas
```
sudo pip3 install pandas
```
#### Matplotlib
```
sudo pip3 install matplotlib
```
#### NLTK
```
sudo pip3 install nltk
```
#### Sklearn
```
sudo pip3 install sklearn
```
#### Gensim
```
sudo pip3 install gensim
```
#### Pacote "punkt" da biblioteca NLTK
```
python3
import nltk
nltk.download('punkt')
nltk.download('stopwords')
exit()
```
#### Python3 tk
```
sudo apt-get install python3-tk
```
#### Django
```
sudo pip3 install django-contrib-comments==1.5
```
<br />


## Download do Word2Vec Pré-treinado (Google News)

https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
<br />
Ou alternativamente:
<br />
https://drive.google.com/file/d/0B_x9Pne58-VTUUM4S1otbUJMclk/view?usp=sharing
<br />
<br />


## Execução
* No arquivo __Experimento.py__, mude as variáveis que contêm os __diretórios__ utilizados, além de mudar o tipo de __separador__ e a __codificação__ do arquivo de entrada.

* O arquivo __example.csv__ pode ser utilizado como exemplo de arquivo de entrada para testar a execução. Não é necessário modificar nenhum arquivo além do __Experimento.py__ para execuções simples: ele pode ser alterado seguindo as instruções descritas nele próprio. Para iniciar, deve-se executar no terminal o seguinte comando:

```
python3 Experimento.py
```


