import graphviz

def create_microgpt_diagram():
    dot = graphviz.Digraph('microGPT', format='png')
    dot.attr(rankdir='TB', nodesep='0.3', ranksep='0.3', margin='0.1')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='#e1f5fe',
             fontname='Helvetica', fontsize='9', penwidth='1', color='#0288d1',
             height='0.3', width='0.1')
    dot.attr('edge', fontname='Helvetica', fontsize='7', color='#455a64', penwidth='1', arrowsize='0.6')

    dot.node('wte', 'wte[tok_id]  4x4\u2192[4]')
    dot.node('wpe', 'wpe[pos_id]  4x4\u2192[4]')
    dot.node('add_emb', '+', shape='circle', fillcolor='#fff9c4', color='#fbc02d', width='0.25', height='0.25', fontsize='8')
    dot.node('norm0', 'RMSNorm [4]')

    dot.edges([('wte', 'add_emb'), ('wpe', 'add_emb')])
    dot.edge('add_emb', 'norm0')

    with dot.subgraph(name='cluster_attn') as c:
        c.attr(label='Attention', style='dashed', color='#1976d2', fontsize='8',
               fontname='Helvetica-Bold', bgcolor='#f3e5f5', margin='6')
        c.node('norm1', 'RMSNorm [4]')
        c.node('qkv', 'Q,K,V = Linear(x)  4x4\u2192[4] each')
        c.node('attn', 'Attn: 2 heads x dim 2\nq\u00b7k/\u221a2, softmax, \u00d7v')
        c.node('wo', 'Linear(wo)  4x4  [4]\u2192[4]')
        c.node('add_attn', '+', shape='circle', fillcolor='#fff9c4', color='#fbc02d', width='0.25', height='0.25', fontsize='8')
        c.edge('norm1', 'qkv')
        c.edge('qkv', 'attn')
        c.edge('attn', 'wo')
        c.edge('wo', 'add_attn')

    dot.edge('norm0', 'norm1')
    dot.edge('norm0', 'add_attn', style='dashed', color='#999999')

    with dot.subgraph(name='cluster_mlp') as c:
        c.attr(label='MLP', style='dashed', color='#388e3c', fontsize='8',
               fontname='Helvetica-Bold', bgcolor='#e8f5e9', margin='6')
        c.node('norm2', 'RMSNorm [4]')
        c.node('fc1', 'Linear(fc1)  16x4  [4]\u2192[16]')
        c.node('relu', 'ReLU [16]')
        c.node('fc2', 'Linear(fc2)  4x16  [16]\u2192[4]')
        c.node('add_mlp', '+', shape='circle', fillcolor='#fff9c4', color='#fbc02d', width='0.25', height='0.25', fontsize='8')
        c.edge('norm2', 'fc1')
        c.edge('fc1', 'relu')
        c.edge('relu', 'fc2')
        c.edge('fc2', 'add_mlp')

    dot.edge('add_attn', 'norm2')
    dot.edge('add_attn', 'add_mlp', style='dashed', color='#999999')

    dot.node('head', 'Linear(lm_head)  4x4  [4]\u2192[4] logits')
    dot.edge('add_mlp', 'head')

    return dot

if __name__ == '__main__':
    diagram = create_microgpt_diagram()
    diagram.render('microgpt_figure', cleanup=True)
    diagram.format = 'svg'
    diagram.render('microgpt_figure', cleanup=True)
    print("Saved microgpt_figure.png and microgpt_figure.svg")
