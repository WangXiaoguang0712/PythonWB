# _*_ coding:utf-8 _*_
from transitions import Machine

def example_form_www():
    # 定义一个自己的类
    class Matter(object):
        pass
    model = Matter()
    # 状态定义
    states=['solid', 'liquid', 'gas', 'plasma']
    # 定义状态转移
    # The trigger argument defines the name of the new triggering method
    transitions = [
        {'trigger': 'melt', 'source': 'solid', 'dest': 'liquid' },
        {'trigger': 'evaporate', 'source': 'liquid', 'dest': 'gas'},
        {'trigger': 'sublimate', 'source': 'solid', 'dest': 'gas'},
        {'trigger': 'ionize', 'source': 'gas', 'dest': 'plasma'}]
    # 初始化
    machine = Machine(model=model, states=states, transitions=transitions, initial='solid')
    # Test
    print model.state    # solid
    # 状体转变
    model.melt()
    print model.state   # liquid


class twi_fsm():
    def __init__(self):
        self.amount = 0.0
        self.states = ['idle', 'wait', 'cola', 'change']
        self.state = 'idle'

    def coin_input(self, val):
        if self.amount < 50:
            self.amount += val
            self.state = self.states[1]
            if self.amount >= 50:
                self.state = self.states[2]
        else:
            print('stop! you have input enough coins.')

    def buy(self):
        if self.state == self.states[2]:
            self.state = self.states[3]
            self.amount -= 50
        else:
            print('please input more coins')

    def retrieve_change(self):
        if self.state == self.states[0] or self.amount <= 0:
            print('illegal operate')
        else:
            self.state == self.states[0]
            self.amount = 0

if __name__ == "__main__":
    # example_form_www()
    model = twi_fsm()
    print(model.state)
    model.coin_input(25)
    model.coin_input(25)
    print(model.state)
    model.coin_input(25)
    model.retrieve_change()
    print(model.state)
    model.buy()

    print(model.state)

    print(model.state)