from tkinter import *
from numpy import *
import CART
import matplotlib
matplotlib.use('TKAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def test_widget_text(root):
    myLabel=Label(root,text="hello")
    myLabel.grid()

def reDraw(tolS,tolN):
    reDraw.f.clf()
    reDraw.a=reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN<2:
            tolN=2
        myTree=CART.createTree(reDraw.rawDat,CART.modelLeaf,CART.modelErr,(tolS,tolN))
        yHat=CART.createForeCast(myTree,reDraw.testDat,CART.modelTreeEval)
    else:
        myTree=CART.createTree(reDraw.rawDat,ops=(tolS,tolN))
        yHat=CART.createForeCast(myTree,reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:,0].tolist(),reDraw.rawDat[:,1].tolist(),s=5)
    reDraw.a.plot(reDraw.testDat,yHat,linewidth=2.0)
    reDraw.canvas.show()

def getInputs():
    try:
        tolN=int(tolNentry.get())
    except:
        tolN=10
        print("enter integer for tolN")
        tolNentry.delete(0,END)
        tolNentry.insert(0,'10')
    try:
        tolS=float(tolSentry.get())
    except:
        tolS=1.0
        print("enter float for tolS")
        tolSentry.delete(0,END)
        tolSentry.insert(0,'1.0')
    return tolN,tolS

def drawNewTree():
    tolN,tolS=getInputs()
    reDraw(tolS,tolN)


def main(root):
    
    #插入标签，设定行和列的位置
    Label(root,text="plot place holder").grid(row=0,columnspan=3)

    Label(root,text="tolN").grid(row=1,column=0)
    #文本输入框
    global tolNentry
    tolNentry=Entry(root)
    tolNentry.grid(row=1,column=1)
    tolNentry.insert(0,'10')

    Label(root,text="tolS").grid(row=2,column=0)
    global tolSentry
    tolSentry=Entry(root)
    tolSentry.grid(row=2,column=1)
    tolSentry.insert(0,'1.0')
    Button(root,text="ReDraw",command=drawNewTree).grid(row=1,column=2,rowspan=3)

   
    global chkBtnVar
    #按钮数形值
    chkBtnVar=IntVar()
    chkBtn=Checkbutton(root,text="model tree",variable=chkBtnVar)
    chkBtn.grid(row=3,column=0,columnspan=2)

    reDraw.f=Figure(figsize=(5,4),dpi=100)
    reDraw.canvas=FigureCanvasTkAgg(reDraw.f,master=root)
    reDraw.canvas.show()
    reDraw.canvas.get_tk_widget().grid(row=0,columnspan=3)

    reDraw.rawDat=mat(CART.loadDataSet('sine.txt'))
    reDraw.testDat=arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)
    reDraw(1.0,10)

if __name__=='__main__':
    root=Tk()
    main(root)
    root.mainloop()
