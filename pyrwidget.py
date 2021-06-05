import numpy as np
import ipywidgets as widgets
from ipywidgets import HBox, VBox
from matplotlib import pyplot as plt
from pyramids import Pyramid as pyr


class LayerTab():
    def __init__(self,im_in,func,opt=True,show=True):
        self.opt = opt
        self.func = func
        self.im_in = im_in.astype('float32')
#         self.im_max = np.abs(self.im_in).max()
#         self.im_sign = np.sign(self.im_in)
        self.im_out = self.im_in.copy().astype('float32')
        # for display
        self.h = 15
        self.w = self.h * im_in.shape[1] // im_in.shape[0]
        if self.opt:
            self.default_gamma = 1.
            self.default_range = 1.
            self.default_bias = 0.
            self.bias_widget = widgets.FloatText(
                value=self.default_bias,
                min=0.00,
                max=1.0,
                step=0.05,
                description='B:',
                readout_format='.2f',
            )
            self.gamma_widget = widgets.FloatText(
                value=self.default_gamma,
                min=0.00,
                max=10.0,
                step=0.05,
                description='gamma:',
                readout_format='.2f',
            )
            self.range_widget = widgets.FloatText(
                value=self.default_range,
                min=0,
                max=3,
                step=0.05,
                description='A:'
            )
            self.layer_button = widgets.Button(
                description='Update layer',
            )
            self.layer_button.on_click(self.update_layer)
            self.reset_button = widgets.Button(
                description='Reset',
            )
            self.reset_button.on_click(self.reset)
        self.out = widgets.Output()
        if self.opt:
            self.tab = VBox(children=[self.bias_widget,
                                    self.gamma_widget,
                                    self.range_widget,
                                    self.layer_button,
                                    self.reset_button,
                                    self.out
                                   ])
        else:
            self.tab = VBox(children=[
                                    self.out
                                   ])
        self.show_images()
        if show:
            display(self.tab)
#     def gain(self,x):
#         g = self.gamma_widget.value
#         return x**g if g>0 else x
#     def apply_gain(self):
#         return self.im_sign*self.gain(np.abs(self.im_in/self.im_max))*self.im_max
#     def weighted_result(self,im0,im):
#         return self.range_widget.value*((1-self.bias_widget.value)*im+self.bias_widget.value*im0.mean())
    def apply_gain(self):
        if self.opt:
            return self.func(self.im_in,self.gamma_widget.value,self.range_widget.value,self.bias_widget.value)
        else:
            return self.im_out
    def show_images(self,fixWindowing=True):
        self.out.clear_output()
        with self.out:
            print('Input range: [%.4f, %.4f]' %(self.im_in.min(),self.im_in.max()))
            print('Output range: [%.4f, %.4f]' %(self.im_out.min(),self.im_out.max()))
            f,ax = plt.subplots(1,2,figsize=(self.w,self.h))
            ax[0].imshow(self.im_in)
#             ax[0].set_axis_off()
            ax[0].set_title('Original', fontsize=24)
            if fixWindowing:
                ax[1].imshow(self.im_out,vmin=self.im_in.min(),vmax=self.im_in.max())
            else:
                ax[1].imshow(self.im_out)
#             ax[1].set_axis_off()
            ax[1].set_title('Processed', fontsize=24)
            plt.tight_layout()
            plt.show()
            if self.opt:
                f,ax = plt.subplots(1,2,figsize=(20,10))
                x = np.linspace(0,1,256)
    #             y = self.weighted_result(x,self.gain(x))
                y = self.func(x,self.gamma_widget.value,self.range_widget.value,self.bias_widget.value)
                ax[0].axis('off')
                ax[1].plot(x,x)
                ax[1].plot(x,y)
                ax[1].grid('on')
                plt.show()
    def update_layer(self,*args):
        if self.opt:
    #         tmp = self.apply_gain()
    #         self.im_out = self.weighted_result(self.im_in,tmp)
            self.im_out = self.apply_gain()
            self.show_images()        
    def reset(self,*args):
        if self.opt:
            self.gamma_widget.value = self.default_gamma
            self.bias_widget.value = self.default_bias
            self.range_widget.value = self.default_range
            self.im_out = self.im_in.copy()
            self.show_images()


        
class MultiLayer():
    def initLayer(self,func,k):
        self.t.append(LayerTab(self.lp[k] if k<len(self.lp) else self.im_in,func,opt=(k<len(self.lp)),show=False))
    def __init__(self,im_in,func,N=None):
        self.h = 15
        self.w = self.h * im_in.shape[1] // im_in.shape[0]
        self.im_in = im_in.copy()
        self.im_out = self.im_in.copy()
        if N is None:
            N = pyr.numlevels(im_in.shape)
        self.lp = pyr.laplacian(im_in,N)
        self.tabs = []
        self.t = []
        for k in range(len(self.lp)+1):
            print('Initialize layer {}...\r'.format(k) if k<len(self.lp) else 'Initialize reconstruction...',end="")
            self.initLayer(func,k)
            self.t[k].update_layer()
            self.tabs.append(self.t[k].tab)
        self.tabs = widgets.Tab(children=self.tabs)
        for k in range(len(self.lp)):
            self.tabs.set_title(k, str(k))
        self.tabs.set_title(len(self.lp),'Reconstruction')
        self.tabs.set_title(len(self.lp)+1,'Parameters')
        self.proc_button = widgets.Button(
            description='Reconstruct',
        )
        self.proc_button.on_click(self.reconstruct)
        self.reset_all_button = widgets.Button(
            description='Reset all layers',
        )
        self.reset_all_button.on_click(self.reset_all)
        self.multitab = VBox(children=[self.tabs, self.proc_button, self.reset_all_button])
        self.reconstruct()
        display(self.multitab)
    def reset_all(self,*args):
        for k in range(len(self.lp)+1):
            self.t[k].reset()
        self.reconstruct()
    def reconstruct(self,*args):
#         for k in range(len(self.lp)):
#             self.t[k].update_layer()
        olp = [self.t[k].im_out.astype("float32") for k in range(len(self.lp))]
        self.im_out = pyr.reconstruct(olp)
        self.t[len(self.lp)].im_out = self.im_out
        self.t[len(self.lp)].show_images(fixWindowing=False)