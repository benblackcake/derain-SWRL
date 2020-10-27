import numpy as np
import glob
import os
from scipy import misc
from skimage.measure import compare_ssim
from skimage.color import rgb2ycbcr, rgb2yuv

from skimage.measure import compare_psnr
from utils import preprocess, downsample, sobel_oper, modcrop, dwt_shape, cany_oper, sobel_direct_oper,\
 batch_Idwt, batch_dwt,dwt_shape, up_sample,up_sample_batch,psnr,calculate_psnr, batch_Swt,batch_ISwt

import tensorflow as tf
import pywt
import cv2

class Benchmark:
    """A collection of images to test a model on."""

    def __init__(self, path_train, path_label, name):
        # self.path_train = path_train
        # self.path_label = path_label

        self.name = name
        # self.images_lr, self.names = self.load_images_by_model(model='LR')
        self.images_rain, self.names = self.load_images_by_model(path_train, model='in')
        self.images_label, self.names = self.load_images_by_model(path_label, model='GT')


    def load_images_by_model(self, path, model, file_format='*'):
        """Loads all images that match '*_{model}.{file_format}' and returns sorted list of filenames and names"""
        # Get files that match the pattern
        filenames = glob.glob(os.path.join(path, '*_' + model + '.' + file_format))
        # Extract name/prefix eg: '/.../baby_LR.png' -> 'baby'
        names = [os.path.basename(x).split('_')[0] for x in filenames]
        return self.load_images(filenames), names

    def load_images(self, images):
        """Given a list of file names, return a list of images"""
        out = []
        for image in images:
            img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB).astype(np.uint8)
            img = modcrop(img,2)
            img = dwt_shape(img)
            out.append(img)

        return out

    def deprocess(self, image):
        """Deprocess image output by model (from -1 to 1 float to 0 to 255 uint8)"""
        image = np.clip(255 * 0.5 * (image + 1.0), 0.0, 255.0).astype(np.uint8)
        return image

    def luminance(self, image):
        # Get luminance
        lum = rgb2ycbcr(image)[:, :, 0]
        # Crop off 4 border pixels
        lum = lum[4:lum.shape[0] - 4, 4:lum.shape[1] - 4]
        # lum = lum.astype(np.float64)
        return lum

    def PSNR(self, gt, pred):
        # gt = gt.astype(np.float64)
        # pred = pred.astype(np.float64)
        # mse = np.mean((pred - gt)**2)
        # psnr = 10*np.log10(255*255/mse)
        # return psnr

        return compare_psnr(gt, pred, data_range=255)

    # def tf_psnr(self, gt, pred, data_range):
    #     psnr = tf.image.psnr(gt, pred, max_val=data_range)

    #     return tf.Session().run(psnr)




    def SSIM(self, gt, pred):
        ssim = compare_ssim(gt, pred, data_range=255, gaussian_weights=True)
        return ssim

    def test_images(self, gt, pred):

        """Applies metrics to compare image lists pred vs gt"""
        avg_psnr = 0
        avg_ssim = 0
        individual_psnr = []
        individual_ssim = []

        print('Length of gt: %s'%len(gt))
        print('Length of pred: %s'%len(pred))
        for i in range(len(pred)):
            # compare to gt
            psnr = self.PSNR(self.luminance(gt[i]), self.luminance(pred[i]))
            ssim = self.SSIM(self.luminance(gt[i]), self.luminance(pred[i]))
            # save results to log_path ex: 'results/experiment1/Set5/baby/1000.png'
            # if save_images:
            #  path = os.path.join(log_path, self.name, self.names[i])
            # gather results
            individual_psnr.append(psnr)
            individual_ssim.append(ssim)
            avg_psnr += psnr
            avg_ssim += ssim

        avg_psnr /= len(pred)
        avg_ssim /= len(pred)
        return avg_psnr, avg_ssim, individual_psnr, individual_ssim

    def validate(self):
        """Tests metrics by using images output by other models"""
        for model in ['bicubic', 'SRGAN-MSE', 'SRGAN-VGG22', 'SRGAN-VGG54', 'SRResNet-MSE', 'SRResNet-VGG22']:
            model_output, _ = self.load_images_by_model(model)
            psnr, ssim, _, _ = self.test_images(self.images_hr, model_output)
            print('Validate %-6s for %-14s: PSNR: %.2f, SSIM: %.4f' % (self.name, model, psnr, ssim))

    def save_image(self, image, path):
        if not os.path.exists(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0])
        misc.toimage(image, cmin=0, cmax=255).save(path)

    def save_images(self, images, log_path, iteration):
        count = 0
        for output, lr, hr, name in zip(images, self.images_rain, self.images_label, self.names):
            # Save output
            path = os.path.join(log_path, self.name, name, '%d_out.png' % iteration)
            self.save_image(output, path)
            # Save output edge map
            # path = os.path.join(log_path, self.name, name, '%d_out_LH.png' % iteration)
            # self.save_image(LH, path)
            # path = os.path.join(log_path, self.name, name, '%d_out_HL.png' % iteration)
            # self.save_image(HL, path)
            # path = os.path.join(log_path, self.name, name, '%d_out_HH.png' % iteration)
            # self.save_image(HH, path)
            # Save ground truth
            path = os.path.join(log_path, self.name, name, '%d_label.png' % iteration)
            self.save_image(hr, path)
            # Save low res
            path = os.path.join(log_path, self.name, name, '%d_rain.png' % iteration)
            self.save_image(lr, path)

            # Hack so that we only do first 14 images in BSD100 instead of the whole thing
            count += 1
            if count >= 14:
                break

    def evaluate(self, sess, y_pred_level_2, y_pred_level_1, log_path=None, iteration=0):
        """Evaluate benchmark, returning the score and saving images."""

        pred = []
        edge_LH = []
        edge_HL = []
        edge_HH = []

        for i, rain in enumerate(self.images_rain):
            # feed images 1 by 1 because they have different sizes
            rain_swt = batch_Swt(rain[np.newaxis], level=2)

            # rain_A = np.stack([rain_dwt[:,:,:,0], rain_dwt[:,:,:,4], rain_dwt[:,:,:,8]], axis=-1)
            # rain_dwt_A_BCD = np.concatenate([rain_dwt[:,:,:,1:4], rain_dwt[:,:,:,5:8], rain_dwt[:,:,:,9:12]], axis=-1)

            rain_swt /= 255.
            # rain_dwt_A_BCD /= 255.

            derain_level_2, derain_level_1 = \
                sess.run([y_pred_level_2, y_pred_level_1], \
                feed_dict={'srresnet_training:0': False,\
                           'LR_SWT:0': rain_swt,\
                            })
            derain_concat = np.concatenate([derain_level_2, derain_level_1], axis=-1)
            # derain_level_2_result = np.concatenate([derain_level_2_LL, derain_level_2_edge], axis=-1)
            # derain_level_1_result = np.concatenate([derain_level_1_LL, derain_level_1_edge], axis=-1)


            # derain_concat = np.concatenate([derain_level_2_result, derain_level_1_result], axis=-1)
            print('__DEBUG__',derain_concat.shape)
            derain_concat *= 255.
            
            derain = batch_ISwt(derain_concat)

            # print('__DEBUG__ Benchmark evaluate', output.shape)
            # print('___debug___')
            # print(output_A[:,:,:,0].shape)
            # print(output_BCD[:,:,:,0:3].shape)
            # print(np.concatenate([np.expand_dims(output_A[:,:,:,0],axis=-1), output_BCD[:,:,:,0:3]], axis=-1).shape)
            # output_A = batch_dwt(output_A)


            # rect_R = np.concatenate([np.expand_dims(output_A[:,:,:,0],axis=-1), output_BCD[:,:,:,0:3]], axis=-1)
            # rect_G = np.concatenate([np.expand_dims(output_A[:,:,:,1],axis=-1), output_BCD[:,:,:,3:6]], axis=-1)
            # rect_B = np.concatenate([np.expand_dims(output_A[:,:,:,2],axis=-1), output_BCD[:,:,:,6:9]], axis=-1)

            # output = np.concatenate([rect_R, rect_G, rect_B], axis=-1)

            # print('__DEBUG__output_shape',output.shape)
            # print('__DEBUG__lr_A_shape',lr_A.shape)

            # lr_A = np.squeeze(lr_A, axis=0)/255.
            # output = np.squeeze(output, axis=0)

            '''__DEBUG__'''
            derain = np.squeeze(derain, axis=0)
            # sr /= np.abs(sr).max()
            # cv2.imshow('__DEBUG__', sr.astype('uint8'))
            # cv2.waitKey(0)
            ''''''
            # derain_A = np.squeeze(derain_A, axis=0)
            # sr_BCD = np.squeeze(sr_BCD, axis=0)



            result = np.clip(derain,0,255).astype(np.uint8) 

            print('__SUCESS__%d'%i)

            # LH = np.abs(sr_BCD[:,:,0])
            # HL = np.abs(sr_BCD[:,:,1])
            # HH = np.abs(sr_BCD[:,:,2])

            # LH *=255.
            # HL *=255.
            # HH *=255.

            # deprocess output
            pred.append(result.astype('uint8'))
            # edge_LH.append(LH.astype('uint8'))
            # edge_HL.append(HL.astype('uint8'))
            # edge_HH.append(HH.astype('uint8'))
        # save images
        if log_path:
            self.save_images(pred, log_path, iteration)
        return self.test_images(self.images_label, pred)
    
    
