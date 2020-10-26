
import tensorflow as tf
import argparse
from benchmark import Benchmark
import os
import sys
from tqdm import tqdm,trange
import matplotlib.pyplot as plt
import srresnet
from utils import (downsample_batch,build_log_dir,preprocess,evaluate_model,batch_bgr2ycbcr,batch_bgr2rgb,batch_dwt,up_sample_batch,
                   get_data_set, sobel_oper_batch, cany_oper_batch, sobel_direct_oper_batch, batch_Swt)
import numpy as np
import pywt
import cv2
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, help='Checkpoint to load all weights from.')
    parser.add_argument('--load-gen', type=str, help='Checkpoint to load generator weights only from.')
    parser.add_argument('--name', type=str, help='Name of experiment.')
    parser.add_argument('--overfit', action='store_true', help='Overfit to a single image.')
    parser.add_argument('--batch-size', type=int, default=16, help='Mini-batch size.')
    parser.add_argument('--log-freq', type=int, default=10000,
                        help='How many training iterations between validation/checkpoints.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for Adam.')
    parser.add_argument('--content-loss', type=str, default='mse', choices=['mse', 'L1','edge_loss_mse','edge_loss_L1'],
                        help='Metric to use for content loss.')
    parser.add_argument('--use-gan', action='store_true',
                        help='Add adversarial loss term to generator and trains discriminator.')
    parser.add_argument('--image-size', type=int, default=96, help='Size of random crops used for training samples.')
    parser.add_argument('--vgg-weights', type=str, default='vgg_19.ckpt',
                        help='File containing VGG19 weights (tf.slim)')
    parser.add_argument('--train-dir', type=str, help='Directory containing training images')
    parser.add_argument('--validate-benchmarks', action='store_true',
                        help='If set, validates that the benchmarking metrics are correct for the images provided by the authors of the SRGAN paper.')
    parser.add_argument('--gpu', type=str, default='0', help='Which GPU to use')
    parser.add_argument('--epoch', type=int, default='1000000', help='How many iterations ')
    parser.add_argument('--is-val', action='store_true', help='How many iterations ')
    parser.add_argument('--upSample', type=int, default='2', help='How much scale ')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    srresnet_training = tf.placeholder(tf.bool, name='srresnet_training')

    srresnet_model = srresnet.Srresnet(training=srresnet_training,\
                              learning_rate=args.learning_rate,\
                              content_loss=args.content_loss,\
                              )

    lr_swt = tf.placeholder(tf.float32, [None, None, None, 24], name='LR_SWT')
    hr_swt = tf.placeholder(tf.float32, [None, None, None, 24], name='HR_SWT')


    sr_pred_level_2, sr_pred_level_1 = srresnet_model.forward(lr_swt)
    # sr_out_pred = srresnet_model.forward_LL_branch(lr_A)
    # sr_BCD_pred = srresnet_model.forward_edge_branch(lr_dwt_edge)

    sr_loss = srresnet_model.loss_function(hr_swt, sr_pred_level_2, sr_pred_level_1)
    sr_opt = srresnet_model.optimize(sr_loss)
    
    '''
    驗證用，input和label的圖要對到。
    '''
    benchmarks = [
        Benchmark('./Benchmarks/Rain12/input', './Benchmarks/Rain12/label', name='Rain12'),
        Benchmark('./Benchmarks/Rain100L/rain', './Benchmarks/Rain100L/norain', name='Rain100L'),
        
    #     #Benchmark('Benchmarks/BSD100', name='BSD100')
    ]


    # Create log folder
    if args.load and not args.name:
        log_path = os.path.dirname(args.load)
    else:
        log_path = build_log_dir(args, sys.argv)

    train_data_path = './dataset/126000ps_24.h5'
    val_data_path = './dataset/112_val_24.h5'
    eval_data_path = './dataset/PreprocessedData_eval_24.h5'


    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        iteration = 0
        epoch = 0

        saver = tf.train.Saver(max_to_keep=100)

        # Load all
        if args.load:
            iteration = int(args.load.split('-')[-1])
            saver.restore(sess, args.load)
            print(saver)
            print("load_process_DEBUG")


        train_data_set = get_data_set(train_data_path,'train')
        train_label_data_set = get_data_set(train_data_path,'label')

        val_data_set = get_data_set(val_data_path,'val')
        val_data_label_set = get_data_set(val_data_path,'label')

        eval_data_set = get_data_set(eval_data_path,'eval')
        eval_data_label_set = get_data_set(eval_data_path,'label')
        #print('eval_data_set',eval_data_set)
        #print('eval_data_label_set',eval_data_label_set)
        val_error_li =[]
        eval_error_li =[]
        fig = plt.figure()
        
        if args.is_val:#暫不用
            benchmarks = [
                Benchmark('Benchmarks/Set5', name='Set5'),
                Benchmark('Benchmarks/Set14', name='Set14'),
                Benchmark('Benchmarks/BSD100', name='BSD100'),
                Benchmark('Benchmarks/UCMerced_LandUse', name='UCMerced_LandUse'),
                Benchmark('Benchmarks/RSSCN7', name='RSSCN7')
            ]

            log_line = ''
            for benchmark in benchmarks:
                psnr, ssim, _, _ = benchmark.evaluate(sess, sr_pred, log_path, iteration)
                print(' [%s] PSNR: %.2f, SSIM: %.4f' % (benchmark.name, psnr, ssim), end='')
                log_line += ',%.7f, %.7f' % (psnr, ssim)
            print()
            # Write to log
            with open(log_path + '/PSNR.csv', 'a') as f:
                f.write(
                    'iteration, set5_psnr, set5_ssim, set14_psnr, set14_ssim, bsd100_psnr, bsd100_ssim,UCMerced_LandUse_psnr, UCMerced_LandUse_ssim,RSSCN7_psnr, RSSCN7_ssim\n'
                 )
                f.write('%d,%s\n' % (iteration, log_line))
          

        else:
            while True:
                t =trange(0, len(train_data_set) - args.batch_size + 1, args.batch_size, desc='Iterations')
        #         #One epoch 
                for batch_idx in t:
                    t.set_description("Training... [Iterations: %s]" % iteration)
                    
        #             # Each 10000 times evaluate model
                    if iteration % args.log_freq == 0:
        # #                 # Loop over eval dataset
        #                 for batch_idx in range(0, len(val_data_set) - args.batch_size + 1, args.batch_size): 
        # #                 # # Test every log-freq iterations
        #                      val_error = evaluate_model(sr_loss, val_data_set[batch_idx:batch_idx + 16], val_data_label_set[batch_idx:batch_idx + 16], \
        #                                                 sess, 112, args.batch_size)
        #                      eval_error = evaluate_model(sr_loss,eval_data_set[batch_idx:batch_idx + 16], eval_data_label_set[batch_idx:batch_idx + 16],\
        #                                                 sess, 112, args.batch_size)
        # #                 val_error_li.append(val_error)
        # #                 eval_error_li.append(eval_error)

        # #                 # # Log error
        # #                 # plt.plot(val_error_li)
        # #                 # plt.savefig('val_error.png')
        # #                 # plt.plot(eval_error_li)
        # #                 # plt.savefig('eval_error.png')
        # #                 # # fig.savefig()

        #                 #print('[%d] Test: %.7f, Train: %.7f' % (iteration, val_error, eval_error), end='')
        #                  # Evaluate benchmarks
        #                 log_line = ''
                        for benchmark in benchmarks:
                            psnr, ssim, _, _ = benchmark.evaluate(sess, sr_pred_level_2, sr_pred_level_1 , log_path, iteration)
                            
        #                     print(' [%s] PSNR: %.2f, SSIM: %.4f' % (benchmark.name, psnr, ssim), end='')
        #                     log_line += ',%.7f, %.7f' % (psnr, ssim)
        # #                 # # print()
        # #                 # # # Write to log
        #                 with open(log_path + '/loss.csv', 'a') as f:
        #                     f.write('%d, %.15f, %.15f%s\n' % (iteration, val_error, eval_error, log_line))
        # #                 # # Save checkpoint
                        saver.save(sess, os.path.join(log_path, 'weights'), global_step=iteration, write_meta_graph=False)
                    
                    # Train SRResnet   
                    batch_rain = train_data_set[batch_idx:batch_idx + 16]
                    batch_label = train_label_data_set[batch_idx:batch_idx + 16]

                    #__DEBUG__
                    # for i in range(batch_rain.shape[0]):
                    #     cv2.imshow('__rain', batch_rain[i])
                    #     cv2.imshow('__label', batch_label[i])

                    #     cv2.waitKey(0)
                    # ycbcr_batch = batch_bgr2ycbcr(batch_hr)
                    batch_rain = batch_bgr2rgb(batch_rain)
                    batch_label = batch_bgr2rgb(batch_label)
                    # batch_lr = downsample_batch(batch_hr, factor=4)

                    batch_swt_rain = batch_Swt(batch_rain, level=2)
                    batch_swt_label = batch_Swt(batch_label, level=2)

                    # batch_dwt_lr[:,:,:,0] /= np.abs(batch_dwt_lr[:,:,:,0]).max()*255.
                    # batch_dwt_lr[:,:,:,4] /= np.abs(batch_dwt_lr[:,:,:,4]).max()*255.
                    # batch_dwt_lr[:,:,:,8] /= np.abs(batch_dwt_lr[:,:,:,8]).max()*255.
                    # batch_dwt_rain_A = np.stack([batch_dwt_rain[:,:,:,0], batch_dwt_rain[:,:,:,4], batch_dwt_rain[:,:,:,8]], axis=-1)
                    # batch_dwt_label_A = np.stack([batch_dwt_label[:,:,:,0], batch_dwt_label[:,:,:,4], batch_dwt_label[:,:,:,8]], axis=-1)

                    batch_swt_rain /= 255.
                    batch_swt_label /= 255.


                    # batch_rain_BCD = np.concatenate([batch_dwt_rain[:,:,:,1:4], batch_dwt_rain[:,:,:,5:8], batch_dwt_rain[:,:,:,9:12]], axis=-1)
                    # batch_label_BCD = np.concatenate([batch_dwt_label[:,:,:,1:4], batch_dwt_label[:,:,:,5:8], batch_dwt_label[:,:,:,9:12]], axis=-1)
                    # batch_lr_BCD = np.concatenate([up_sample_batch(batch_dwt_lr_A[:,:,:,1:4], factor=2),\
                    #                                up_sample_batch(batch_dwt_lr_A[:,:,:,5:8], factor=2),\
                    #                                up_sample_batch(batch_dwt_lr_A[:,:,:,9:12], factor=2)], axis=-1)
                    # batch_lr = downsample_batch(batch_hr, factor=4)
                    # batch_lr_BCD = up_sample_batch(batch_lr_BCD, factor=2)

                    # batch_rain_BCD = batch_rain_BCD/255.
                    # batch_label_BCD = batch_label_BCD/255.

                    # batch_label =batch_label / 255.

                    _, err = sess.run([sr_opt,sr_loss],\
                         feed_dict={srresnet_training: False,\
                                    lr_swt:batch_swt_rain,\
                                    hr_swt:batch_swt_label,\
                                    })

                    #print('__training__ %s' % iteration)
                    iteration += 1
                print('__epoch__: %s' % epoch)
                epoch += 1

if __name__ == "__main__":
    main()
