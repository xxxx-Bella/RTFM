import matplotlib.pyplot as plt
import os

def visulization(item, smooth=False, window_size=5):
    breakpoint()
    scene = item.key()
    num_frame = item.value()[0]
    abnormal_index_list = item.value()[1:]
    frame_index = list(num_frame)
    breakpoint()
    
    # if smooth:
    #     pred = smooth_curve(pred, window_size)

    plt.figure(figsize=(10, 6))
    plt.plot(frame_index, item, color='#ff7f0e', linewidth=2)
    
    # plt.title('Prediction Scores vs Frame Index', fontsize=16)
    plt.xlabel('Frame Index', fontsize=14)
    plt.ylabel('Prediction Score', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True)
    # plt.legend()
    # output_path = os.path.join(log_dir, f'epoch{epoch}-{scene}-score.png')
    # plt.savefig(output_path)
    plt.show()
    plt.close()




scene_dic = {'bike': [[11216], [0, 270], [6000, 6300], [7300, 7452]],
            'cross': [[10336], [4854, 5112, 0.81], [5323, 5672, 0.82], [7083, 7784, 0.73]],
            'farm': [[5280], [342, 600, 0.68], [342, 600, 0.68], [4708, 5122, 0.85]],
            # 'highway': [[5472], []],
            # 'railway': [[882], []],
            # 'solar': [[2408], []],
            # 'vehicle': [[2584], []]
            }
for item in scene_dic:
    visulization(item)

# scene_dic0 = {'bike': [18427, (0, 40)],
#        'cross': 6244,
#        'farm':2387,
#        'highway': 2820,
#         'railway': 882,
#         'solar': 2450,
#         'vehicle': 2643
#         }