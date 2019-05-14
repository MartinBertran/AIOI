import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from keras.utils import to_categorical


from keras.utils import Sequence


class BatchGeneratorFilterNew(Sequence):

    def __init__(self, dataframe, training=True, secret_list=[], utility_list=[], output_list=[], input_list=[], batch_size=16,
                     img_size=(64, 64, 1), shuffling=True, output_type='binary', n_secrets=1,n_utility=1, secret_prior=None, utility_prior=None, randomize_images=True):
        
        #possible outputs in list = 's','sp','u','up','pt' (secret, secret_prior, utility, utility_prior, passthrough'
        self.output_list = output_list        
        self.training_data = training
        self.randomize_images=randomize_images

        self.df = dataframe.copy().dropna()
        self.img_size = img_size

        self.output_type = output_type

        self.filepath = self.df['filepath'].tolist()
        self.secret=self.df[secret_list].as_matrix()
        self.utility = self.df[utility_list].as_matrix()

        # possible extra inputs, image is always the first input
        self.extra_inputs=None
        if len(input_list)>0:
            self.extra_inputs=self.df[input_list].as_matrix()

        self.utility_p= np.array(utility_prior)
        self.secret_p = np.array(secret_prior)

        self.n_secrets = n_secrets
        self.n_utilities = n_utility

        self.shuffling = shuffling
        self.batch_size = batch_size
        self.cur_index = 0

        self.train_datagen = ImageDataGenerator(
            rotation_range=0,  # 15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0,  # 0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest')

        # Free up memory of unneeded data
        del dataframe
        self.df = None
        del self.df
        self.shuffle()
    
    def shuffle(self):
        if self.extra_inputs is not None:
            self.filepath,self.secret,self.utility, self.extra_inputs = shuffle(self.filepath,self.secret,self.utility, self.extra_inputs)
        else:
            self.filepath,self.secret,self.utility = shuffle(self.filepath,self.secret,self.utility)

    
    def __len__(self):
        return int(np.ceil(len(self.filepath) / float(self.batch_size)))

    def __getitem__(self, idx):
        
        batch_img = self.filepath[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_utility = self.utility[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_secret = self.secret[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.extra_inputs is not None:
            batch_extra_inputs = self.extra_inputs[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_extra_inputs = np.concatenate([np.stack(batch_extra_inputs[:,i]) for i in np.arange(batch_extra_inputs.shape[-1])], axis=-1)


        X_data = np.zeros([len(batch_img), self.img_size[0], self.img_size[1], self.img_size[2]])

        # load img 
        for i in np.arange(X_data.shape[0]):
            if self.img_size[2] > 1:
                grayscale = False
            else:
                grayscale = True
            img = image.load_img(batch_img[i],
                                 target_size=(self.img_size[0], self.img_size[1]),
                                 interpolation='bicubic', grayscale=grayscale)
            img = image.img_to_array(img)
            if self.randomize_images:
                img = self.train_datagen.random_transform(img)
            img /= 255
            X_data[i, ...] = img


        if self.output_type=='binary':
            Y_secret = np.array(batch_secret)
            Y_utility = np.array(batch_utility)
        else:
            Y_secret = to_categorical(np.array(batch_secret), num_classes=self.n_secrets)
            Y_utility = to_categorical(np.array(batch_utility), num_classes=self.n_utilities)
            
            Y_utility_p = np.zeros([len(batch_img),self.n_utilities])
            Y_utility_p[:] = self.utility_p
            
        Y_secret_p = np.zeros([len(batch_img),self.n_secrets])
        Y_secret_p[:] = self.secret_p


        inputs = [X_data]

        if self.extra_inputs is not None:
            inputs += [batch_extra_inputs]

        outputs=[]
        for o in self.output_list:
            if o =='s':
                outputs +=[Y_secret]
            elif o=='sp':
                outputs += [Y_secret_p]
            elif o == 'u':
                outputs += [Y_utility]
            elif o=='up':
                outputs += [Y_utility_p]
            elif o=='pt':
                outputs +=[X_data]
            else:
                print("Unrecognized output in output list")
                return
            
        #print('generator yielded a batch %d' % idx)

        return (inputs, outputs)
    





