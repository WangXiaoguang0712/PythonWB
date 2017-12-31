#coding:utf-8
__author__ = 'T'
import pygame as pg
import  sys
import copy
import random
#主角登场
class Hero(object):
    def __init__(self):
        self.id = pg.image.load('tortoise.bmp')
        self.rect = self.id.get_rect()
        self.pos = self.rect.move((100,2*self.rect.height))
        self.initpos = copy.copy(self.pos)
        self.speed = [0,0]
        self.jumplimit = 2 * self.rect.height

    def jump(self):
        self.speed[1] = -8
#坏蛋登场
class Badguy(object):
    def __init__(self,pleft,ptop):
        self.id = pg.image.load('fireball.bmp')
        self.rect = self.id.get_rect()
        self.pos = self.id.get_rect().move((pleft,ptop - self.rect.height))
        self.speed = [-4,0]

    @classmethod
    def generate(cls,pleft = 800,ptop = 400):
        return cls(pleft,ptop);

class Timer(object):
    def __init__(self):
        self.t = 0
        self.lvl = 0

    def tick(self):
        self.t = self.t - 1 if self.t > 0 else 0

#初始化
pg.init()
pg.display.set_caption('小乌龟')
size = width,height = 800,600
screen = pg.display.set_mode(size)
bg_color = 255,255,255
showtime = 10,40

#主角登场
no1 = Hero()
#第一个坏蛋登场
no2 = Badguy.generate(pleft = width,ptop = no1.rect.height * 3)
#计时器
timer = Timer()
timer.t = random.randrange(showtime[0],showtime[1])

while True:
    #判断退出事件
    for event in pg.event.get():
        if event.type == pg.QUIT:
            sys.exit()
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_UP:
                no1.jump()

    #当上一个坏蛋死亡，准备孕育下一个坏蛋
    if no2.pos.right < 0:
        timer.tick()
        if timer.t == 0 :
            no2 = Badguy.generate(pleft = width,ptop = no1.rect.height * 3)
            timer.t = random.randrange(showtime[0],showtime[1])
            timer.lvl += 1
            no2.speed[0] -= timer.lvl

    #碰撞检测
    if no1.pos.left < no2.pos.left < no1.pos.right and no1.pos.bottom > no2.pos.top > no1.pos.top:
        sys.exit()

    #移动图形
    no1.pos = no1.pos.move(no1.speed)
    no2.pos = no2.pos.move(no2.speed)

    #判断方向
    if no1.pos.left < 0 or  no1.pos.right > width:
        no1.speed[0] = -no1.speed[0]
        #水平翻转图像
        no1.id = pg.transform.flip(no1.id,True,False)
    if no1.pos.top <= no1.initpos.top - no1.jumplimit:
        no1.speed[1] = - no1.speed[1]
    if no1.pos.top >= no1.initpos.top:
        no1.speed[1] = 0
    #填充背景
    screen.fill(bg_color)
    #双缓冲
    #更新图像
    screen.blit(no1.id,no1.pos)#bilt方法将一个图像覆盖到另一个位置上
    screen.blit(no2.id,no2.pos)
    #更新界面
    pg.display.flip()
    #延迟10毫秒
    pg.time.delay(10)

