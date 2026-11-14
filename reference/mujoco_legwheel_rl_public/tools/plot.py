import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

def test0():
    def softsign(x):
        return x / (1 + np.abs(x))

    # 生成输入范围
    x = np.linspace(-10, 10, 500)
    y_softsign = softsign(x)
    y_sin = np.sin(x)
    y_cos = np.cos(x) 
    y_tanh = np.tanh(x)

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_softsign, label='softsign(x)', linewidth=2.5)
    plt.plot(x, y_sin, label='sin(x)', linestyle='--')
    plt.plot(x, y_cos, label='cos(x)', linestyle=':')
    plt.plot(x, y_tanh, label='tanh(x)', linestyle='-.')

    plt.title('Activation Functions Comparison', fontsize=14)
    plt.xlabel('Input Value (x)', fontsize=12)
    plt.ylabel('Output Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend(fontsize=12, loc='upper right')

    # 标记softsign关键点
    plt.scatter([-1, 0, 1], [softsign(-1), softsign(0), softsign(1)],
                color='red', zorder=5)
    plt.text(-1, softsign(-1)-0.1, '(-1,-0.5)', ha='center')
    plt.text(0, softsign(0)+0.1, '(0,0)', ha='center')
    plt.text(1, softsign(1)+0.1, '(1,0.5)', ha='center')

    plt.ylim(-1.2, 1.2)
    plt.show()


def test_exp(): # draw exp(-x/sigma) function, sigma=linspace(0.1, 1, 10)
    x = np.linspace(0, 1, 100)
    startposx = len(x)//10
    stepposx = 8
    sigma = np.linspace(0.1, 1, 10)
    plt.figure(figsize=(10, 6))
    for i in range(len(sigma)):
        y = np.exp(-x/sigma[i])
        plt.plot(x, y, label=f'sigma={sigma[i]:.1f}')
        # 在曲线末端添加文本标签
        showposx = min(startposx+i*stepposx-3, len(x)-1)
        showposy = min(startposx+i*stepposx, len(y)-1)
        plt.text(x[showposx], y[showposy], f' σ={sigma[i]:.1f}', fontsize=10)
    plt.ylim(0, 1.2)
    plt.xlim(0, 1)
    plt.legend()
    plt.grid()
    plt.title('exp(-x/sigma)')
    # plt.show()
    plt.savefig('exp.png')

def test_exp_square(): # draw exp(-x/sigma) function, sigma=linspace(0.1, 1, 10)
    x = np.linspace(0, 1, 100)
    startposx = len(x)//10
    stepposx = 8
    sigma = np.linspace(0.1, 1, 10)
    plt.figure(figsize=(10, 6))
    for i in range(len(sigma)):
        y = np.exp(-x**2/sigma[i])
        plt.plot(x, y, label=f'sigma={sigma[i]:.1f}')
        # 在曲线末端添加文本标签
        showposx = min(startposx+i*stepposx-3, len(x)-1)
        showposy = min(startposx+i*stepposx, len(y)-1)
        plt.text(x[showposx], y[showposy], f' σ={sigma[i]:.1f}', fontsize=10)
    plt.ylim(0, 1.2)
    plt.xlim(0, 1)
    plt.legend()
    plt.grid()
    plt.title('exp(-x^2/sigma)')
    # plt.show()
    plt.savefig('exp_square.png')

def test_abs_exp(): # draw exp(-abs(x)/sigma) function, sigma=linspace(0.1, 1, 10), x = linspace(-1, 1, 100)
    x = np.linspace(0, 2, 100)
    startposx = len(x)//10
    stepposx = 8
    sigma = np.linspace(0.1, 1, 10)
    plt.figure(figsize=(10, 6))
    for i in range(len(sigma)):
        y = np.exp(-np.abs(x)/sigma[i])
        plt.plot(x, y, label=f'sigma={sigma[i]:.1f}')
        # 在曲线末端添加文本标签
        showposx = min(startposx+i*stepposx-3, len(x)-1)
        showposy = min(startposx+i*stepposx, len(y)-1)
        plt.text(x[showposx], y[showposy], f' σ={sigma[i]:.1f}', fontsize=10)
    plt.ylim(0, 1.2)
    plt.xlim(0, 2)
    plt.legend()
    plt.grid()
    plt.title('exp(-abs(x)/sigma)')
    # plt.show()
    plt.savefig('abs_exp.png')


if __name__ == '__main__':
    # test0()
    # test_exp()
    # test_exp_square()
    test_abs_exp()