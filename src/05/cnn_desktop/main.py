from tkinter import *
from tkinter import filedialog
from PIL import ImageDraw, Image, ImageGrab
import numpy as np
from skimage import color, io
import os
from cnn_class import getData, trainModel, loadModel

class Paint(object):
    # 類別初始化函數
    def __init__(self):
        self.root = Tk()
        self.root.title('手寫阿拉伯數字辨識')

        #defining Canvas
        self.c = Canvas(self.root, bg='white', width=280, height=280)
        
        self.image1 = Image.new('RGB', (280, 280), color = 'white')
        self.draw = ImageDraw.Draw(self.image1) 

        self.c.grid(row=1, columnspan=6)

        # 建立【辨識】按鈕
        self.classify_button = Button(self.root, text='辨識', command=lambda:self.classify(self.c))
        self.classify_button.grid(row=0, column=0, columnspan=2, sticky='EWNS')

        # 建立【清畫面】按鈕
        self.clear = Button(self.root, text='清畫面', command=self.clear)
        self.clear.grid(row=0, column=2, columnspan=2, sticky='EWNS')

        # 建立【存檔】按鈕
        self.savefile = Button(self.root, text='存檔', command=self.savefile)
        self.savefile.grid(row=0, column=4, columnspan=2, sticky='EWNS')

        # 建立【預測】文字框
        self.label1 = Label(self.root, height=2, width=10, text='預測：')
        self.label1.grid(row=2, column=3, columnspan=2)
        self.prediction_text = Text(self.root, height=2, width=10)
        self.prediction_text.grid(row=2, column=4, columnspan=2)
        
        # 定義滑鼠事件處理函數
        self.setup()
        
        # 監聽事件
        self.root.mainloop()

    # 滑鼠事件處理函數
    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 15
        self.color = 'black'
        
        # 定義滑鼠事件處理函數，包括移動滑鼠及鬆開滑鼠按鈕
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    # 移動滑鼠 處理函數
    def paint(self, event):
        paint_color = self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            # 顯示設定>100%，抓到的區域會變小
            # 畫圖同時寫到記憶體，避免螢幕字型放大，造成抓到的畫布區域不足
            self.draw.line((self.old_x, self.old_y, event.x, event.y), fill='black', width=self.line_width)

        self.old_x = event.x
        self.old_y = event.y

    # 鬆開滑鼠按鈕 處理函數
    def reset(self, event):
        self.old_x, self.old_y = None, None

    # 【清畫面】處理函數
    def clear(self):
        self.c.delete("all")
        self.image1 = Image.new('RGB', (280, 280), color = 'white')
        self.draw = ImageDraw.Draw(self.image1) 
        self.prediction_text.delete("1.0", END)

    # 【存檔】處理函數
    def savefile(self):
        f = filedialog.asksaveasfilename( defaultextension=".png", 
            filetypes = [("png file",".png")])
        # asksaveasfile return `None` if dialog closed with "cancel".
        if f is None: 
            return
        self.image1.save(f)

    # 【辨識】處理函數
    def classify(self, widget):
        img = self.image1.resize((28, 28)).convert('L')      
        img = np.array(img)
        img = (255 - img) / 255
        
        img2=Image.fromarray(img) 
        img = np.reshape(img, (1, 28, 28, 1))
        
        # Predict digit
        pred = model.predict([img], verbose=False)
        # Get index with highest probability
        pred = np.argmax(pred)
        self.prediction_text.delete("1.0", END)
        self.prediction_text.insert(END, pred)


if __name__ == '__main__':
    # 訓練模型或載入既有的模型
    if(os.path.exists('mnist_model.keras')):
        print('load model ...')
        model = loadModel()
    else:
        print('train model ...')
        X_train, y_train, X_test, y_test = getData()
        model = trainModel(X_train, y_train, X_test, y_test)

    print(model.summary())
    
    # 顯示視窗
    Paint()

