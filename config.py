class Config(object):

    '''train set'''
    train_data_split = 'training'
    val_data_split = 'validation'
    test_data_split = 'testing'

    learn_rate=0.0001
    train_batch_size = 1

    seed = 123
    num_threads = 6
    max_epochs = 50


    '''save setting'''
    save_checkpoint_every = 1  # num of epoch to save model
    checkpoint_path = './checkpoint'  # model save path
    start_from = ''#'/storage/mgwang/video_grounding/DVSA-pytorch/checkpoint/cvpr.pth'  # if it is not None,will restore the model,and training from this model.
    result_save_path = './result'  # testing submission result to save
    test_from = ''  # testing model path
