import os
import shutil

delete_patterns = ['vis_train', 'vis_val', 
                   'checkpoint.pth.tar', 'checkpoint.pth.tar_best_mae', 'checkpoint.pth.tar_best_mse',
                   'checkpoint.pth.tar_best_mae_before', 'checkpoint.pth.tar_best_mse_before']

def rm_dir_or_file(path):
    print('remove:' + path)
    if os.path.isfile(path):
        os.remove(path)
    if os.path.isdir(path):
        shutil.rmtree(path) 

def delete_exp_data(root_dir):
    
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        for child_item in os.listdir(item_path):
            child_item_path = os.path.join(item_path, child_item)
            
            if child_item in delete_patterns:
                rm_dir_or_file(child_item_path)
            
            if child_item.startswith('checkpoint.pth.tar_best_mae_before_Ep_') or \
                child_item.startswith('checkpoint.pth.tar_best_mse_before_Ep_'):
                    rm_dir_or_file(child_item_path)
                
            if child_item.startswith('events.out.tfevents.'):
                rm_dir_or_file(child_item_path)
                
            if child_item.startswith('Ep_') and child_item.endswith('.pth') :
                new_child_item_path = child_item_path + '.info'
                print('created:' + new_child_item_path)
                with open(new_child_item_path, 'w'):
                    pass
                
                rm_dir_or_file(child_item_path)

if __name__ == "__main__":
    root_dir = 'exp/sim_match/SHHA_Sim_Match_LevelMap/MocHRBackbone_hrnet48'
    delete_exp_data(root_dir=root_dir)
    