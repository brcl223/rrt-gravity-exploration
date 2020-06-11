from sys import argv, exit

import numpy as np
import matplotlib.pyplot as plt

NUM_ELEMENTS = 25
DATA = {
    'rewards-ppo': {
        'filename': 'rewards-ppo.png',
        'Joint Torques': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Cumulative Reward',
            'title': 'DDPG JT Cumulative Reward',
            'filename': 'ppo-jt_nog-final-rewards.txt',
            'plotpos': 111
        },
        'Joint Torques w/ GComp': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Cumulative Reward',
            'title': 'DDPG JT w/ Gravity Cumulative Reward',
            'filename': 'ppo-jt_g-final-rewards.txt',
            'plotpos': 111
        },
        'PD': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Cumulative Reward',
            'title': 'DDPG PD Cumulative Reward',
            'filename': 'ppo-pd_nog-final-rewards.txt',
            'plotpos': 111
        },
        'PD w/ GComp': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Cumulative Reward',
            'title': 'PPO Controllers Cumulative Reward',
            'filename': 'ppo-pd_g-final-rewards.txt',
            'plotpos': 111
        }
    },
    'length-ppo': {
        'filename': 'length-ppo.png',
        'Joint Torques': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Average Episode Length',
            'title': 'DDPG JT Episode Length',
            'filename': 'ppo-jt_nog-final-length.txt',
            'plotpos': 111
        },
        'Joint Torques w/ GComp': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Average Episode Length',
            'title': 'DDPG JT Episode Length',
            'filename': 'ppo-jt_g-final-length.txt',
            'plotpos': 111
        },
        'PD': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Average Episode Length',
            'title': 'DDPG JT Episode Length',
            'filename': 'ppo-pd_nog-final-length.txt',
            'plotpos': 111
        },
        'PD w/ GComp': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Average Episode Length',
            'title': 'PPO Average Episode Length',
            'filename': 'ppo-pd_g-final-length.txt',
            'plotpos': 111
        },
    },
    'hits-ppo': {
        'filename': 'hits-ppo.png',
        'Joint Torques': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Targets Hit per Episode',
            'title': 'DDPG JT Target Count',
            'filename': 'ppo-jt_nog-final-hits.txt',
            'plotpos': 111
        },
        'Joint Torques w/ GComp': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Targets Hit per Episode',
            'title': 'DDPG JT Target Count',
            'filename': 'ppo-jt_g-final-hits.txt',
            'plotpos': 111
        },
        'PD': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Targets Hit per Episode',
            'title': 'DDPG JT Target Count',
            'filename': 'ppo-pd_nog-final-hits.txt',
            'plotpos': 111
        },
        'PD w/ GComp': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Targets Hit per Episode',
            'title': 'DDPG Average Target Count',
            'filename': 'ppo-pd_g-final-hits.txt',
            'plotpos': 111
        },
    },
    'rewards-jt': {
        'filename': 'rewards-jt.png',
        'Joint Torques': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Cumulative Reward',
            'title': 'DDPG JT Cumulative Reward',
            'filename': 'ddpg-jt_nog-final-rewards.txt',
            'plotpos': 111
        },
        'Joint Torques w/ GComp': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Cumulative Reward',
            'title': 'DDPG JT w/ Gravity Cumulative Reward',
            'filename': 'ddpg-jt_g-final-rewards.txt',
            'plotpos': 111
        },
    },
    'rewards': {
        'filename': 'rewards.png',
        # 'PD Acc Pen': {
        #     'xrange_step': 1,
        #     'xlabel': 'Episode',
        #     'ylabel': 'Cumulative Reward',
        #     'title': 'DDPG JT Cumulative Reward',
        #     'filename': 'ddpg-pd_nog-pen_acc-final-rewards.txt',
        #     'plotpos': 111
        # },
        # 'PD Acc Pen w/ GComp': {
        #     'xrange_step': 1,
        #     'xlabel': 'Episode',
        #     'ylabel': 'Cumulative Reward',
        #     'title': 'DDPG JT w/ Gravity Cumulative Reward',
        #     'filename': 'ddpg-pd_g-pen_acc-final-rewards.txt',
        #     'plotpos': 111
        # },
        'PD': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Cumulative Reward',
            'title': 'DDPG PD Cumulative Reward',
            'filename': 'ddpg-pd_nog-final-rewards.txt',
            'plotpos': 111
        },
        'PD w/ GComp': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Cumulative Reward',
            'title': 'DDPG Controllers Cumulative Reward',
            'filename': 'ddpg-pd_g-final-rewards.txt',
            'plotpos': 111
        }
    },
    'length': {
        'filename': 'length.png',
        'Joint Torques': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Average Episode Length',
            'title': 'DDPG JT Episode Length',
            'filename': 'ddpg-jt_nog-final-length.txt',
            'plotpos': 111
        },
        'Joint Torques w/ GComp': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Average Episode Length',
            'title': 'DDPG JT Episode Length',
            'filename': 'ddpg-jt_g-final-length.txt',
            'plotpos': 111
        },
        'PD': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Average Episode Length',
            'title': 'DDPG JT Episode Length',
            'filename': 'ddpg-pd_nog-final-length.txt',
            'plotpos': 111
        },
        'PD w/ GComp': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Average Episode Length',
            'title': 'DDPG Average Episode Length',
            'filename': 'ddpg-pd_g-final-length.txt',
            'plotpos': 111
        },
    },
    'hits': {
        'filename': 'hits.png',
        'Joint Torques': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Targets Hit per Episode',
            'title': 'DDPG JT Target Count',
            'filename': 'ddpg-jt_nog-final-hits.txt',
            'plotpos': 111
        },
        'Joint Torques w/ GComp': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Targets Hit per Episode',
            'title': 'DDPG JT Target Count',
            'filename': 'ddpg-jt_g-final-hits.txt',
            'plotpos': 111
        },
        'PD': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Targets Hit per Episode',
            'title': 'DDPG JT Target Count',
            'filename': 'ddpg-pd_nog-final-hits.txt',
            'plotpos': 111
        },
        'PD w/ GComp': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Targets Hit per Episode',
            'title': 'DDPG Average Target Count',
            'filename': 'ddpg-pd_g-final-hits.txt',
            'plotpos': 111
        },
    },
    'reward-iters': {
        'filename': 'reward-iters.png',
        # 'Joint Torques': {
        #     'xrange_step': 1,
        #     'xlabel': 'Episode',
        #     'ylabel': 'Targets Hit per Episode',
        #     'title': 'DDPG JT Target Count',
        #     'filename': 'ddpg-jt_nog-final-hits.txt',
        #     'plotpos': 111
        # },
        # 'JT w/ GComp': {
        #     'xrange_step': 1,
        #     'xlabel': 'Episode',
        #     'ylabel': 'Targets Hit per Episode',
        #     'title': 'DDPG JT Target Count',
        #     'filename': 'ddpg-jt_g-final-hits.txt',
        #     'plotpos': 111
        # },
        'PD': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Targets Hit per Episode',
            'title': 'DDPG JT Target Count',
            'filename': 'ddpg-pd_nog-final-reward-iters.txt',
            'plotpos': 111
        },
        'PD w/ GComp': {
            'xrange_step': 1,
            'xlabel': 'Episode',
            'ylabel': 'Targets Hit per Episode',
            'title': 'DDPG Average Target Count',
            'filename': 'ddpg-pd_g-final-reward-iters.txt',
            'plotpos': 111
        },
    },
    'testloss': {
        'filename': 'testloss.png',
        'testloss': {
            'xrange_step': 2,
            'xlabel': 'Test Point',
            'ylabel': 'G(q\') Error',
            'title': 'G(q) NN Error',
            'filename': 'test-errors.txt',
            'plotpos': 111,
        }
    },
    'avgloss': {
        'filename': 'avgloss.png',
        'avgloss': {
            'xrange_step': 300,
            'xlabel': 'Training Iteration',
            'ylabel': 'Mean Loss at Iteration',
            'title': 'Average Loss During Training',
            'filename': 'avg-loss.txt',
            'plotpos': 111
        }
    },
    'meanvars': {
        'filename': 'meanvars.png',
        'mean': {
            'xrange_step': 300,
            'xlabel': 'Training Iteration',
            'ylabel': 'Max Loss at Iteration',
            'title': 'Max Loss During Training',
            'filename': 'max-loss.txt',
            'plotpos': 211
        },
        'var': {
            'xrange_step': 300,
            'xlabel': 'Training Iteration',
            'ylabel': 'Max Variance at Iteration',
            'title': 'Max Variance During Training',
            'filename': 'max-var.txt',
            'plotpos': 212,
        }
    }
}


def check_name(name):
    if name in DATA:
        return True
    return False


def main():
    valid_use_names = ' | '.join(DATA.keys())

    if len(argv) != 2:
        print(f"{argv[0]} expects one input: <name>")
        print(f"Potential options: [{valid_use_names}]")
        print("Exiting...")
        exit(1)

    name = argv[1]
    if not check_name(name):
        print(f"{name} is an invalid selection. Expecting one of [{valid_use_names}]")
        print("Exiting...")
        exit(1)

    metadata = DATA[name]

    fig = plt.figure()

    for k, v in metadata.items():
        if type(v) is not dict:
            continue

        print(f"Current: {k}")
        data = np.loadtxt(f"data/{v['filename']}")
        ax = fig.add_subplot(v['plotpos'])
        step = len(data) // NUM_ELEMENTS
        new_data = np.zeros(NUM_ELEMENTS)
        for i in range(NUM_ELEMENTS):
            current_start = i * step
            new_data[i] = np.average(data[current_start:current_start+step])

        x_vals = np.arange(0, NUM_ELEMENTS*step*v['xrange_step'],
                           step*v['xrange_step'])

        ax.plot(x_vals, new_data, label=k)
        # data[data<-100]=-100
        # ax.plot(data, alpha=0.2)
        # ax.legend()
        ax.title.set_text(v['title'])
        plt.ylabel(v['ylabel'])
        plt.xlabel(v['xlabel'])

    plt.savefig(f"data/{metadata['filename']}")


if __name__ == '__main__':
    main()
