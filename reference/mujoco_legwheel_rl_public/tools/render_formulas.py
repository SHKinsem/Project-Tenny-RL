import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'path'  # 文字转路径，无字体依赖

formulas = {
    'f_gamma':  r'$\gamma = 0.5^{\,t_{ctrl}\,/\,t_{half}}$',
    'f_ri':     r'$r_i = \exp\left(-|\mathrm{err}_i|\,/\,\sigma_i\right)$',
    'f_R':      r'$R = \sum_{i \in +} w_i\, r_i \;+\; \sum_{j \in -} w_j\, p_j, \qquad w_j < 0$',
    'f_S':      r'$S(\theta) = \sum_i \, w_i^{base}\, R_i(\theta)$',
    'f_anchor': r'$\forall i:\; w_i \leftarrow c \cdot w_i \;\Longleftrightarrow\; r_s \leftarrow c \cdot r_s \quad (\mathrm{policy\ unchanged})$',
    'f_Ei':     r'$E_i = \dfrac{2}{N}\left(\sum_{run:\,x_i=+1} y_{run} \;-\; \sum_{run:\,x_i=-1} y_{run}\right)$',
    'f_hb':     r'$N_k = N_0 \cdot (1/3)^{k}$',
    'f_mo':     r'$\min_{\theta}\; \left(\,-T(\theta),\; P(\theta)\,\right), \qquad T = \sum_{i \in +} w_i^{base} R_i, \quad P = \sum_{j \in -} |w_j^{base} R_j|$',
    'f_delta':  r'$\Delta = \overline{S_{base}} - \overline{S_c} \;>\; 2\,\sigma_{pool} = 2\sqrt{\sigma_c^2/3 + \sigma_{base}^2/3} \qquad (S = \mathrm{RMSE},\ \downarrow\ \mathrm{better})$',
    'f_SE':     r'$\mathrm{SE}(E_i) = \dfrac{2\sigma_{noise}}{\sqrt{N}} \qquad |E_i| < 2\,\mathrm{SE}(E_i) = \dfrac{4\sigma_{noise}}{\sqrt{N}} \;\Rightarrow\; \mathrm{indistinguishable}$',
}

for name, tex in formulas.items():
    fig = plt.figure(figsize=(0.1, 0.1))
    t = fig.text(0, 0, tex, fontsize=16, color='#222222')
    fig.canvas.draw()
    bbox = t.get_window_extent()
    w, h = bbox.width / fig.dpi, bbox.height / fig.dpi
    plt.close(fig)
    fig = plt.figure(figsize=(w + 0.1, h + 0.1))
    fig.text(0.5 / (w + 0.1), 0.5 / (h + 0.1) * 0.5, tex, fontsize=16, color='#222222')
    fig.savefig(f'docs/images/formulas/{name}.svg', format='svg',
                bbox_inches='tight', pad_inches=0.05, transparent=True,
                metadata={'Date': None})  # 去掉 dc:date,重渲染不再产生幻影 diff
    plt.close(fig)
    print('rendered', name)
