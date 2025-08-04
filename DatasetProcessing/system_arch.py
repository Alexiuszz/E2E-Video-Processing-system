from graphviz import Digraph

# Create a single architecture diagram
dot = Digraph(comment="System Architecture", format='png')
dot.attr(rankdir='TB', fontsize='10')

# ASR Phase
dot.node('ASR', 'ASR Phase')
dot.node('V', 'Raw Video')
dot.node('F', 'FFmpeg\n(Audio Extraction)')
dot.node('E', 'Enhancement Modules\n• noisereduce\n• DeepFilterNet2\n• Silero VAD')
dot.node('A', 'ASR Engines\n• Whisper\n• NeMo\n• OpenAI Whisper API')
dot.node('T', 'Transcript (.txt)')

# Topic Segmentation
dot.node('Seg', 'Topic Segmentation')
dot.node('S1', 'Sentence Tokeniser')
dot.node('S2', 'SBERT Encoder')
dot.node('S3', 'Similarity Engine')
dot.node('S4', 'Segmenter\n• Adaptive Text Tiled \n• Simple Depth-based boundary detection')
dot.node('S5', 'Topic Merger')
dot.node('S6', 'BERTopic Labeller')
dot.node('J', 'Segmented JSON')

# API Phase
dot.node('API', 'API Gateway')
dot.node('U', 'POST /upload')
dot.node('TR', 'POST /transcribe')
dot.node('SEG', 'POST /segment')

# Flow: ASR Phase
dot.edge('ASR', 'V')
dot.edge('ASR', 'T')
dot.edge('V', 'F')
dot.edge('F', 'E')
dot.edge('E', 'A')
dot.edge('A', 'T')

# Flow: Topic Segmentation
dot.edge('Seg', 'T')
dot.edge('Seg', 'J')
dot.edge('T', 'S1')
dot.edge('S1', 'S2')
dot.edge('S2', 'S3')
dot.edge('S3', 'S4')
dot.edge('S4', 'S5')
dot.edge('S5', 'S6')
dot.edge('S6', 'J')

# Flow: API Integration
dot.edge('API', 'U')
dot.edge('API', 'TR')
dot.edge('API', 'SEG')
dot.edge('U', 'F')
dot.edge('TR', 'A')
dot.edge('SEG', 'S1')

# Render
dot.render('system_architecture', view=True)