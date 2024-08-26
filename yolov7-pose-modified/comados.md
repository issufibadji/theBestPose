# PyTorch VS  TensorFlow

<p align="center">
   Framework PyTorch VS  TensorFlow usando com YOLOv7<br>
    <br><table>
    <thead>
        <tr>
            <th align="center">
                <img width="20" height="1"> 
                <p>
                    <small>FEATURES</small>
                </p>
            </th>
            <th align="center">
                <img width="300" height="1"> 
                <p> 
                    <small>
                        PYTORCH
                    </small>
                </p>
            </th>
            <th align="left">
                <img width="140" height="1">
                <p align="left"> 
                    <small>
                   TENSORFLOW 
                    </small>
                </p>
            </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Flexibilidade </td>
            <td>Gráficos de execução dinâmica</td>
            <td>Gráficos de execução estática</td>
        </tr>
         <tr>
            <td>Facilidade de Uso </td>
            <td>Curva de aprizagem amigavél</td>
            <td>Curva de aprizagem menos amigavél</td>
         </tr>
         <tr>
            <td>Comunidade/Suporte </td>
            <td>Acadêmico</td>
            <td>Industrial</td>
        </tr>
         <tr>
            <td>Velocidade de Prototipagem </td>
            <td>Preferido para prototipagem rápida de modelos</td>
            <td>Possui um ecossistema robusto(impl. produçao)</td>
        </tr>
         <tr>
            <td>Suporte GPU /CPU </td>
            <td> GPU e CPU </td>
            <td> GPU e CPU</td>
        </tr>
    </tbody>
</table></p>

**Resumindo**: Ambos os frameworks são poderosos e têm comunidades ativas. Se você está começando e caso você não tenha uma GPU disponível, ambos os frameworks podem ser executados em CPU, mas o treinamento de modelos em CPUs pode ser significativamente mais lento em comparação com GPUs. deseja uma curva de aprendizado mais suave, PyTorch pode ser uma escolha sólida.

# YOLOv7 com PyTorch

Para executar YOLOv7 com PyTorch usando uma GPU, você precisará atender a alguns pré-requisitos. Aqui estão os principais passos e requisitos para configurar seu ambiente:

1. **Hardware**:
    - **GPU**: Você precisará de uma GPU NVIDIA compatível com CUDA para acelerar o treinamento e a inferência do YOLOv7. GPUs mais poderosas geralmente proporcionam um treinamento mais rápido.

2. **Software**:
    - **Sistema Operacional**: O PyTorch com suporte a GPU é compatível com sistemas operacionais Windows, Linux e macOS, mas o Linux é frequentemente preferido para tarefas de aprendizado profundo devido à estabilidade e ao suporte CUDA.
    - **visual studio**: Certifique-se de ter o visual studio instalado (https://visualstudio.microsoft.com/pt-br/visual-cpp-build-tools/)
    - **Python**: Certifique-se de ter o Python instalado. É recomendável usar o Python 3.x, como Python 3.6 ou superior.
         ```
         python -v
         ```
      Se não usa ```pip3``` ou ```Anaconda``` para instalar [Anaconda](https://www.anaconda.com/download/)
    - 
    - **PyTorch**: Instale o PyTorch com suporte a CUDA, que é necessário para aproveitar a GPU. Você pode instalar o PyTorch usando pip ou conda. Por exemplo:
      ```
      pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
      ```
       Acesse: [Pytorch.org](https://pytorch.org/get-started/locally/)
3. **CUDA Toolkit**:
    - Você precisa instalar o NVIDIA CUDA Toolkit compatível com a versão do PyTorch que você está usando. Certifique-se de instalar a versão correta para a sua GPU. Você pode baixar o CUDA Toolkit no site da NVIDIA.

      Acesse: [Developer.nvidia](https://developer.nvidia.com/cuda-downloads)
4. **cuDNN**:
    - Instale a biblioteca cuDNN (Deep Neural Network Library) compatível com a versão do CUDA Toolkit. Ela é fundamental para o desempenho das operações de deep learning em GPUs NVIDIA.

   Acesse: [developer.cudnn](https://developer.nvidia.com/rdp/cudnn-archive)

5. **Configuração do Windows**:

   Consulte os requisitos de hardware e os de software listados acima. Leia o guia de instalação da CUDA® para Windows.
   Verifique se os pacotes de software da NVIDIA instalados correspondem às versões listadas acima. Em particular, o TensorFlow não carregará sem o arquivo cuDNN64_8.dll. Para usar uma versão diferente, consulte o guia sobre criar da origem para o Windows.
   Adicione os diretórios de instalação da CUDA, CUPTI e cuDNN à variável de ambiente %PATH%. Por exemplo, se o CUDA® Toolkit estiver instalado em C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0 e cuDNN em C:\tools\cuda, atualize seu %PATH% para corresponder a:

   ```
   SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin;%PATH%
   SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\extras\CUPTI\lib64;%PATH%
   SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include;%PATH%
   SET PATH=C:\tools\cuda\bin;%PATH%
   ``` 
6. **Ambiente Virtual (Opcional)**:
    - É recomendável criar um ambiente virtual Python, como o Anaconda, para isolar seu ambiente de desenvolvimento e facilitar a gerência de pacotes.
     ```
    conda env create -f environment.py  
     ```
7. **Código e Modelos YOLOv7**:
    - Clone o repositório oficial do YOLOv7 no GitHub (https://github.com/WongKinYiu/yolov7/tree/pose) ou (https://github.com/WongKinYiu/yolov7/releases) para obter o código e os modelos pré-treinados.
      - Depois coloco o peso yolo7 dentro de rais do projeto e execute seguintes comandos:
         ```
           cd projeto/yolov7-pose
        ```
        ```
           conda create -n yolov7_pose python=3.9
        ```
        ```
          conda activate yolov7_pose
        ```

8. **Dependências adicionais**:
    - Você também precisará de outras bibliotecas Python específicas para o YOLOv7, como NumPy, OpenCV, Pillow e outras que podem ser especificadas no arquivo `requirements.txt` no repositório do YOLOv7.
        ```
        pip install -r requirements.txt 
      ```
9. **importaçao torch**
   
   ```
   import torch
   ```
     ```
      print(torch.__version__) //Saida:  version torch
     ```
     ```
      torch.cuda.is_available() // saida: True
     ```
     ```
      torch.cuda.current_device() // saida: 0
     ```
     ```
     torch.cuda.get_device_name(0) // saida: NVIDIA GeForce RTX 3070 Ti Laptop GPU
     ```

10. **Execute Pose Estimation**:

   ``` python run_pose.py  –-source 0 ```

    Para excutar inference de  video:
   
     ``` python run_pose.py  –-source [path to video]```
   
    Para excutar com GPU:
   
      ``` python run_pose.py  –-source 0  –-device 0 ```

Depois de atender a esses pré-requisitos, você deve ser capaz de configurar e executar o YOLOv7 com PyTorch na sua GPU. Lembre-se de seguir as instruções específicas fornecidas no repositório do YOLOv7 para treinar ou executar a detecção de objetos, pois pode haver configurações e comandos específicos para o modelo. Certifique-se de ler a documentação e os tutoriais relacionados para um guia passo a passo.

# YOLOv7 com TensorFlow
Executar o YOLOv7 com TensorFlow usando uma GPU requer a configuração de um ambiente apropriado. Aqui estão os principais pré-requisitos e passos necessários para executar o YOLOv7 com TensorFlow na GPU:

1. **Hardware**:
   - **GPU NVIDIA**: Você precisará de uma GPU NVIDIA compatível com CUDA para aproveitar o treinamento e a inferência acelerados por GPU. Certifique-se de ter uma GPU adequada ao seu projeto, com suporte ao CUDA Toolkit.

2. **Software**:
   - **Sistema Operacional**: O TensorFlow é compatível com sistemas operacionais Windows, Linux e macOS. Linux é geralmente preferido para desenvolvimento de deep learning devido à estabilidade e ao suporte ao CUDA.

   - **Python**: É necessário ter o Python instalado no seu sistema. Recomenda-se usar o Python 3.x, como Python 3.6 ou superior.
     ```
     python -v
     ```
   - **TensorFlow**: Instale o TensorFlow-GPU, que é a versão do TensorFlow otimizada para uso com GPUs NVIDIA. Você pode instalá-lo usando pip:
     ```
     pip install tensorflow-gpu
     ``` 
     ```
     pip install tensorflow==1.15      # CPU
     pip install tensorflow-gpu==1.15  # GPU
     ```
    ````
     python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ````
   
3. **CUDA Toolkit**:
   - Você precisa instalar o NVIDIA CUDA Toolkit compatível com a versão do TensorFlow que você está usando. Baixe a versão correta do CUDA Toolkit no site da NVIDIA.
     
   - Acesse: [Developer.nvidia](https://developer.nvidia.com/cuda-downloads)
4. **cuDNN**:
   - Instale a biblioteca cuDNN (Deep Neural Network Library) compatível com a versão do CUDA Toolkit que você instalou. Isso é fundamental para otimizar as operações de deep learning na GPU.

    Acesse: [developer.cudnn](https://developer.nvidia.com/rdp/cudnn-archive)

6. **Ambiente Virtual (Opcional)**:
   - É recomendável criar um ambiente virtual Python, como o Anaconda, para isolar seu ambiente de desenvolvimento e facilitar a gerência de pacotes.
    ```
    conda env create -f environment.py  
     ```
7. **Código e Modelos YOLOv7**:
   - Você precisará obter a implementação do YOLOv7 que seja compatível com TensorFlow. Lembre-se de que YOLOv7 é geralmente implementado em PyTorch, então você pode precisar procurar por implementações específicas para TensorFlow ou fazer a conversão a partir da implementação original em PyTorch.
   - Clone o repositório oficial do YOLOv7 no GitHub (https://github.com/WongKinYiu/yolov7/tree/pose) ou (https://github.com/WongKinYiu/yolov7/releases) para obter o código e os modelos pré-treinados.
      - Depois coloco o peso yolo7 dentro de rais do projeto e execute seguintes comandos:
         ```
           cd projeto/yolov7-pose
        ```
        ```
           conda create -n yolov7_pose python=3.9
        ```
        ```
          conda activate yolov7_pose
        ```
8. **Dependências adicionais**:
   - Dependendo da implementação específica do YOLOv7 em TensorFlow, você pode precisar de outras bibliotecas Python, como NumPy, OpenCV, Pillow e outras que podem ser especificadas no arquivo `requirements.txt` ou no README do projeto.
     ```
     pip install -r requirements.txt
      ```
9. **Configurações Específicas**:
   - Siga as instruções específicas fornecidas na documentação ou no repositório do YOLOv7 em TensorFlow para configurar seu ambiente e executar o treinamento ou a detecção de objetos. Pode haver comandos e configurações específicas que variam de uma implementação para outra.

10. **Execute Pose Estimation**:

   ``` python run_pose.py  –-source 0 ```
   
   Para excutar inference de  video:
   
   ``` python run_pose.py  –-source [path to video]```
   
   Para excutar com GPU:
   
   ``` python run_pose.py  –-source 0  –-device 0 ```
Após atender a esses pré-requisitos e configurar seu ambiente, você deve ser capaz de executar o YOLOv7 com TensorFlow na sua GPU NVIDIA. Certifique-se de ler a documentação e os tutoriais relacionados ao projeto específico que você está usando, pois pode haver diferenças entre as implementações em termos de configurações e comandos específicos.
