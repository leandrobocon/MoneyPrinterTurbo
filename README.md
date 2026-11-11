<div align="center">

# MoneyPrinterTurbo 💸

### Gerador Completo de Vídeos Curtos com Inteligência Artificial

Basta fornecer um **tema** ou **palavras-chave** para gerar automaticamente o roteiro, narrar o áudio, selecionar ou criar os vídeos com IA, sincronizar legendas, mixar música de fundo e renderizar o vídeo final em alta definição.

[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](https://github.com/leandrobocon/MoneyPrinterTurbo)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)

[Português](README.md) | [English](README-en.md) | [Repositório Original (Upstream)](https://github.com/harry0703/MoneyPrinterTurbo)

</div>

## 🚀 Novidades e Modificações Deste Fork

Este repositório traz melhorias essenciais de sincronização narrativa, variedade visual dinâmica e otimização para hardware Apple Silicon (Mac MPS):

### 1. 🎯 Sincronização Inteligente de Cenas por Conteúdo Narrativo
- **Distribuição Proporcional por Conteúdo:** A duração de cada cena é calculada dinamicamente com base na narração real em áudio e nas legendas (`compute_content_scene_durations`).
- **Eliminação de Descompasso:** Em vídeos com múltiplos tópicos, as cenas permanecem na tela exatamente pelo tempo em que seu assunto está sendo narrado, em vez de ficarem presas a repetições globais desincronizadas.
- **Suporte a Timestamps nos Prompts:** Permite definir marcações temporais diretas no prompt (ex: `Cena 1 (0:00 - 0:25): ...`) ou balancear automaticamente por orações e parágrafos do roteiro.

### 2. 🎬 Repetições Dinâmicas com Perspectivas Alternadas (Reverso e Zoom)
Quando o tempo de narração de uma cena for maior que o clipe gerado (ex: clipe de 10s para 25s de fala), o sistema não faz apenas um corte repetido monótono:
- **1ª Repetição (Efeito Reverso / Ping-Pong):** O vídeo é reproduzido de trás para frente suavemente (`reverse`), criando continuidade natural a partir do ponto onde o movimento parou.
- **2ª Repetição (Zoom Focal / Punch-in 1.25x):** Um corte aproximado focado no centro ou no terço superior da ação destaca detalhes e simula um novo enquadramento de câmera.
- **Repetições Subsequentes:** Alterna reversões com zoom e variações de ângulo, mantendo o dinamismo visual ao longo de toda a narração.

### 3. ⚡ Otimização do MiniMax H3 Local e ComfyUI (Apple Silicon / Mac MPS)
- **Seletor de Resolução no WebUI:** Escolha entre `768x448` (padrão nítido e rápido), `832x480` (480p) e `576x320`.
- **Proteção Preventiva Float8:** Detecção e alerta automático bloqueando modelos FP8/Float8 (como Wan 14B) no Mac, pois o PyTorch Metal (MPS) não oferece suporte a esse formato, direcionando para modelos otimizados (MiniMax H3 int8 ou Wan 1.3B bf16).

### 4. 🔊 Estabilidade no Edge TTS e Vinhetas
- **Streaming de Áudio Longo:** Extensão do timeout com renovação de deadline por pacote recebido, evitando abortar narrações longas ou conexões lentas.
- **Mixagem Robusta de Vinhetas:** Geração de trilha silenciosa (`anullsrc`) ao concatenar vinhetas de abertura ou fechamento que não possuem faixa de áudio nativa, evitando erros de renderização no FFmpeg.

---

## 🖥️ Visão Geral e Interface

<h4 align="center">WebUI</h4>

![](docs/webui-en.jpg)

<h4 align="center">API</h4>

![](docs/api.jpg)

---

## 🎯 Principais Recursos

### Fluxo de Criação
- Interface WebUI amigável via Streamlit e API RESTful integrada via FastAPI.
- Geração automática de ponta a ponta: do tema ao roteiro, narração, materiais visuais, legendas e edição final.
- Histórico de tarefas, reprocessamento individual e exportação de vídeos.

### Roteiro e Modelos de Linguagem (LLMs)
- Suporte a OpenAI (GPT-4o, etc.), Anthropic Claude, Google Gemini, DeepSeek, Kimi / Moonshot AI, Qwen, Ollama local e endpoints compatíveis com OpenAI.
- Geração de roteiros em múltiplos idiomas ou entrada de roteiro personalizado.

### Materiais Visuais e Vídeos
- **Geração Local com IA:** ComfyUI com Wan 2.1 e MiniMax H3 Local.
- **APIs de Vídeo e Imagem:** Metaso MiniMax H3, Wavespeed, Volcano Engine Seedance, MuAPI e modelos Text-to-Image.
- **Bancos Gratuitos de Vídeo:** Pexels, Pixabay e Coverr.
- **Upload de Materiais Locais:** Suporte para imagens e vídeos do usuário.

### Voz, Legendas e Música de Fundo
- **Edge TTS Gratuito:** Vozes neurais de alta qualidade em português (como `pt-BR-AntonioNeural` e `pt-BR-FranciscaNeural`) e diversas outras línguas sem necessidade de chave de API.
- **Legendas Automáticas:** Alinhamento de legendas por frase ou palavra, personalização de fonte, cor, tamanho, contorno e posição.
- **Música de Fundo (BGM):** Trilha sonora personalizada, aleatória ou via IA, com controle independente de volume.

### Formatos de Saída
- Retrato `9:16 (1080×1920)` para TikTok, Instagram Reels e YouTube Shorts.
- Paisagem `16:9 (1920×1080)` para YouTube tradicional.
- Quadrado `1:1 (1080×1080)`.

---

## 📦 Instalação e Execução

### Pré-requisitos
- Python 3.11 ou superior
- FFmpeg instalado no sistema

### 1. Clonar o Repositório
```shell
git clone https://github.com/leandrobocon/MoneyPrinterTurbo.git
cd MoneyPrinterTurbo
```

### 2. Criar Ambiente Virtual e Instalar Dependências

Com **uv** (recomendado):
```shell
uv python install 3.11
uv sync --frozen
```

Ou com **venv + pip**:
```shell
python3.11 -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Iniciar a Aplicação

#### Iniciar o WebUI:
```shell
# macOS / Linux
./webui.sh

# Windows
.\webui.bat
```
Ou diretamente via Python:
```shell
streamlit run ./webui/Main.py --browser.serverAddress=127.0.0.1 --server.enableCORS=True --browser.gatherUsageStats=False
```

Acesse o navegador em: [http://127.0.0.1:8501](http://127.0.0.1:8501)

#### Iniciar a API:
```shell
python main.py
```
Documentação da API disponível em: [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs)

---

## 🐳 Execução via Docker

```shell
docker compose -f docker-compose.release.yml up
```

---

## 📄 Licença

Este projeto é distribuído sob os termos da licença MIT. Consulte o arquivo [LICENSE](LICENSE) para obter mais informações.
