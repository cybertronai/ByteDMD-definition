import graphviz

def create_microgpt_diagram():
    # Initialize the directed graph
    dot = graphviz.Digraph('microGPT', format='png')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.6', ranksep='0.8')

    # Global node styles
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='#e1f5fe',
             fontname='Helvetica', penwidth='1.5', color='#0288d1')
    dot.attr('edge', fontname='Helvetica', fontsize='10', color='#455a64', penwidth='1.5')

    # --- INPUTS ---
    dot.node('token_id', 'Token ID', fillcolor='#f5f5f5', shape='ellipse')
    dot.node('pos_id', 'Position ID', fillcolor='#f5f5f5', shape='ellipse')

    # --- EMBEDDINGS ---
    dot.node('wte', 'Token Embedding\n(wte)')
    dot.node('wpe', 'Positional Embedding\n(wpe)')
    dot.node('add_emb', 'Add (+)', shape='circle', fillcolor='#fff9c4', color='#fbc02d')
    dot.node('emb_norm', 'RMSNorm\n(Pre-Norm)')

    dot.edges([('token_id', 'wte'), ('pos_id', 'wpe')])
    dot.edges([('wte', 'add_emb'), ('wpe', 'add_emb')])
    dot.edge('add_emb', 'emb_norm')

    # --- ATTENTION BLOCK ---
    with dot.subgraph(name='cluster_attn') as c:
        c.attr(label='Attention Block', style='dashed', color='#1976d2',
               fontname='Helvetica-Bold', bgcolor='#f3e5f5')

        c.node('attn_norm', 'RMSNorm')
        c.node('qkv', 'Linear Projections\n(wq, wk, wv)')
        c.node('attn_core', 'Self-Attention\n(Single Position: softmax(q\xb7k/\u221ad) = 1.0)')
        c.node('attn_wo', 'Linear Projection\n(wo)')
        c.node('add_attn', 'Add (+)', shape='circle', fillcolor='#fff9c4', color='#fbc02d')

        c.edge('attn_norm', 'qkv')
        c.edge('qkv', 'attn_core', label=' q, k, v')
        c.edge('attn_core', 'attn_wo', label=' x_attn')
        c.edge('attn_wo', 'add_attn')

    # Connections to/around Attention Block
    dot.edge('emb_norm', 'attn_norm')
    dot.edge('emb_norm', 'add_attn', style='dashed', label=' x_residual') # Residual connection

    # --- MLP BLOCK ---
    with dot.subgraph(name='cluster_mlp') as c:
        c.attr(label='MLP Block', style='dashed', color='#388e3c',
               fontname='Helvetica-Bold', bgcolor='#e8f5e9')

        c.node('mlp_norm', 'RMSNorm')
        c.node('fc1', 'Linear\n(mlp_fc1)')
        c.node('relu', 'ReLU\n(x * (1 if x > 0 else 0))')
        c.node('fc2', 'Linear\n(mlp_fc2)')
        c.node('add_mlp', 'Add (+)', shape='circle', fillcolor='#fff9c4', color='#fbc02d')

        c.edge('mlp_norm', 'fc1')
        c.edge('fc1', 'relu')
        c.edge('relu', 'fc2')
        c.edge('fc2', 'add_mlp')

    # Connections to/around MLP Block
    dot.edge('add_attn', 'mlp_norm')
    dot.edge('add_attn', 'add_mlp', style='dashed', label=' x_residual') # Residual connection

    # --- OUTPUT ---
    dot.node('lm_head', 'Linear\n(lm_head)')
    dot.node('logits', 'Logits', fillcolor='#f5f5f5', shape='ellipse')

    dot.edge('add_mlp', 'lm_head')
    dot.edge('lm_head', 'logits')

    return dot

if __name__ == '__main__':
    print("Generating microGPT architecture diagram...")
    diagram = create_microgpt_diagram()

    output_path = 'microgpt_figure'
    diagram.render(output_path, cleanup=True)

    # Also render SVG
    diagram.format = 'svg'
    diagram.render(output_path, cleanup=True)

    print(f"Saved {output_path}.png and {output_path}.svg")
