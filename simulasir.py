import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from enum import Enum, auto
from random import uniform, random
from math import cos, sin, pi, inf, sqrt
import os


# スクリプトのあるディレクトリパス（ここは適宜書き換えてください）.
this_dir = "C:\\Users\\tamaki_py\\Desktop\\Python\\SIR"
# 粒子半径.
radius = 0.075
# 時間刻み.
dt = 0.01
# 正方形 Box の一辺の長さの半分.
half = 5
# 速度の大きさ.
speed = 1
# 回復にかかる時間（回復率 γ の逆数）.
time_to_recover = 2
# バッファ.
buff = 300


# 図.
fig = plt.figure(figsize=plt.figaspect(1/1))
# 図の背景色は黒.
fig.set_facecolor('black')


class State(Enum):
    """
    状態（State）を表す.

    Susceptible:
        未感染状態.
    Infected:
        感染状態.
    Recovered:
        免疫獲得状態.
    """
    Susceptible = auto()
    Infected = auto()
    Recovered = auto()


class Person:

    """
    人間を表すクラス.

    Attributes
    ----------
    co: np.ndarray
        現在の 2 次元座標.
    coords: np.ndarray
        過去の座標を全て記憶.
    arg: float
        初期速度の偏角.
    standstill: bool
        停止しているか否か.
    vel: np.ndarray
        現在の速度（2 次元）.
    state: State
        現在の感染状態.
    infection_start: int
        感染が始まった時間の index.
    """

    def __init__(
        self,
        co,
        state,
        standstill
    ):
        """
        Parameters
        ----------
        co: np.ndarray
            初期 2 次元座標.
        state: State
            初期感染状態.
        standstill: bool
            停止させるか否か.
        """
        self.co = co
        self.coords = np.array([self.co])
        self.arg = uniform(-pi, pi)
        self.standstill = standstill
        if self.standstill:
            # 停止させるのであれば初期速度は (0.0, 0.0).
            vel = np.array([0.0, 0.0])
        else:
            # そうでないのであれば偏角（self.arg）方向に大きさ speed で等速運動.
            vel = np.array([speed * cos(self.arg), speed * sin(self.arg)])
        self.vel = vel
        self.state = state
        self.infection_start = inf

    def move(
        self
    ):
        """
        もし self が停止しているのならば, 動かす.
        """
        if self.standstill:
            self.vel = np.array([speed * cos(self.arg), speed * sin(self.arg)])


class Population:

    """
    人口集団を表すクラス.

    Attributes
    ----------
    number: int
        総人口数.
    R0: float
        基本再生産数.
        全ての粒子が一定の速度（speed）で運動するときの平均自由行程の考え方を利用.
        2 つの粒子間の相対速度ベクトル V の大きさ |V| は,
        一方の速度ベクトルを v , 他方のを v' とし, v と v' のなす角を θ とすると,
        余弦定理より |V|^2 = |v|^2 + |v'|^2 - 2 |v||v'|cosθ.
        ここで粒子間の相対速度ベクトル V の大きさの平均値の 2 乗を考える.
        cosθ の平均値は 0 , |v| = |v'| = speed なので,
        それは 2 * (speed ** 2) となる.
        よって, 各粒子間の相対速度の大きさの平均値は √2 * speed となるので,
        感染者が回復するまでにかかる時間 time_to_recover までに,
        粒子の半径を radius として, 粒子間の距離が 2 * radius の間にある粒子とは衝突することになる.
        正方形 Box の一辺の長さの半分は half なので,
        粒子一つが占める平均面積は 4 * (half ** 2) を総人口数 number で割ったものになる.
        よって, {2 * (2 * radius)} * {(√2 * speed) * time_to_recover} = R0 * {4 * (half ** 2)} / number
        となって, R0 = radius * (√2 * speed) * time_to_recover * number / (half ** 2) となる.
    persons: list
        人間（Person インスタンス）のリスト.
    p: float
        行動制限の割合.
        None の場合には行動制限を行わない.
    Re: float
        実効再生産数.
        p の行動制限では本来,
        行動制限する人間（粒子）をシミュレーションから除くべきだが,
        今回は視覚的な分かりやすさのため停止させている.
        それにより, 単純に Re = (1 - p) * R0 とはならない.
        一つの粒子のみが運動し他が停止している粒子の平均自由行程の考え方を利用すると,
        そのときの粒子間の相対速度の大きさの平均値は speed になるので,
        このときの再生産数は radius * speed * time_to_recover * number / (half ** 2) = R0 / √2.
        一つの運動している粒子から見ると,
        (1 - p) の割合で運動している他の粒子との相対速度の大きさの平均値は √2 * speed,
        p の割合で運動を停止している他の粒子との相対速度の大きさの平均値は speed となるので,
        このときの再生産数は (1 - p) * R0 + p * (R0 / √2)
        一つの停止している粒子から見ると,
        (1 - p) の割合で運動している他の粒子との相対速度の大きさの平均値は speed,
        p の割合で運動を停止している他の粒子との相対速度の大きさの平均値は 0 となるので,
        このときの再生産数は (1 - p) * (R0 / √2)
        よってこれらの粒子の再生産数の平均値をとると実効再生産数を得ることができ,
        Re = (1 - p) * {(1 - p) * R0 + p * (R0 / √2)} + p * {(1 - p) * (R0 / √2)}
        = (1 - p) * {1 + (√2 - 1) * p} * R0
    times: np.ndarray
        時間の Numpy 配列.
    zeros: np.ndarray
        ゼロ Numpy 配列.
    I: np.ndarray
        各時間における感染者数の配列.
    S: np.ndarray
        各時間における未感染者数の配列.
    R: np.ndarray
        各時間における免疫獲得者数の配列.
    comeback: bool
        行動制限を解除するか否か.
    """

    def __init__(
        self,
        number,
        comeback,
        p=None
    ):
        """
        Parameters
        ----------
        number: int
            総人口数.
        comeback: bool
            行動制限を解除するか否か.
        p: float
            行動制限の割合.
            None の場合には行動制限を行わない.
        """
        self.number = number
        self.R0 = (radius * (sqrt(2) * speed) * time_to_recover * self.number) / (half ** 2)
        print("R0: %f" % self.R0)
        self.persons = []
        self.p = p
        init_coords = []
        if self.p is not None:
            self.Re = (1.0 - self.p) * (1.0 + (sqrt(2) - 1) * self.p) * self.R0
            print("Re: %f" % self.Re)
        # 総人口数分の人間を生成.
        for i in range(self.number):
            if i == 0:
                # 最初の人間は感染者として初期条件を設定.
                # 初期座標は原点に設定.
                init_co = np.array([0.0, 0.0])
                infected = Person(co=init_co,
                                  state=State.Infected,
                                  standstill=False)
                infected.infection_start = 0
                self.persons.append(infected)
            else:
                # それ以外の人間は未感染者として初期条件を設定.
                while True:
                    # 以前の人間の初期座標とは異なるまで while ループで座標をランダムに生成.
                    init_co = np.array([uniform(-half, half), uniform(-half, half)])
                    appropriate = True
                    for pre_init_co in init_coords:
                        dist = np.linalg.norm(init_co - pre_init_co, ord=2)
                        # 以前の人間の初期座標との距離が 3 *（半径）の時は不適合とする.
                        # 最低 2 *（半径）で判定しないと, 人間を表す粒子が重なる.
                        # この判定距離を大きく取りすぎると, 人口数が大きいとき
                        # 新規座標がいつまで経っても不適合となりうることに注意.
                        if dist <= 3 * radius:
                            appropriate = False
                    if appropriate:
                        break
                # 行動制限の割合（p）を指定していない時は全ての人間を動かす.
                # そうでない場合は, 一様ランダムに 0.0 ~ 1.0 の数を生成し,
                # その値が行動制限の割合以下ならば停止させ, そうでない時は動かす.
                if self.p is None:
                    standstill = False
                elif random() < self.p:
                    standstill = True
                else:
                    standstill = False
                self.persons.append(Person(co=init_co,
                                           state=State.Susceptible,
                                           standstill=standstill))
            init_coords.append(init_co)
        self.times = np.array([0])
        self.zeros = np.array([0])
        self.I = np.array([1])
        self.S = np.array([self.number - 1])
        self.R = np.array([0])
        self.comeback = comeback

    def simulate_sir(
        self,
        threshold=2
    ):
        """
        SIR モデルのシミュレーションを行う.

        Parameters
        ----------
        threshold: int
            感染者数がこの threshold（閾値）以下になったとき,
            行動制限を解除する（ただし self.comeback が真であるとき）.
        """
        # 時間の index.
        i = 0
        # 感染者数が閾値（threshold）を超えたか否か.
        exceeded_threshold = False
        # 行動制限を解除したか否か.
        comebacked = False
        # 感染が収束した時の時間のインデックス.
        subsided = None
        # 感染が収束してからバッファの分だけ時間のインデックスが進むまで SIR モデルのシミュレーションを行う.
        while True:
            # 各人間に対し, 感染が始まってから回復時間経った時には,
            # その人間の感染状態を免疫獲得状態に変更する.
            # そして各人間を移動させる.
            # 正方形 Box の壁に当たった場合には反射させる.
            for person in self.persons:
                if (i - person.infection_start) * dt > time_to_recover:
                    person.state = State.Recovered
                # 速度の時間刻み dt のスカラー倍分進む.
                person.co += dt * person.vel
                # x 軸方向での反射.
                if abs(person.co[0]) > half:
                    person.co[0] = np.sign(person.co[0]) * half
                    person.vel[0] *= -1
                # y 軸方向での反射.
                if abs(person.co[1]) > half:
                    person.co[1] = np.sign(person.co[1]) * half
                    person.vel[1] *= -1
            # 各人間の接触を判定する.
            # 人間たちの index を全て集めたものを indices とし,
            # indices の先頭の index を取り出し k とする.
            # その k を index として持つ人間を person とし,
            # 残った indices の各 index を l として,
            # その l を index として持つ人間を other とする.
            # person と other が接触している場合, その 2 人を反射させるが,
            # もし一方が未感染状態で, もう一方が感染状態のときには,
            # 未感染状態の方を感染状態にする.
            indices = list(range(self.number))
            while len(indices) > 0:
                k = indices.pop(0)
                person = self.persons[k]
                for l in indices:
                    other = self.persons[l]
                    vec = person.co - other.co
                    dist = np.linalg.norm(vec, ord=2)
                    # person と other の距離が 2 *（半径）より小さいとき接触したと判定.
                    # 簡単のため 3 人以上の同時接触は考えない.
                    if 0 < dist < 2 * radius:
                        normal = vec / dist
                        # person と other が重ならないようにするための補正（correction）.
                        correction = (radius - (dist / 2)) * normal
                        person.co += correction
                        other.co -= correction
                        # 物理的な完全弾性衝突の場合は -= np.dot(person.vel - other.vel, normal) * normal.
                        person.vel -= (2 * np.dot(person.vel, normal)) * normal
                        # 物理的な完全弾性衝突の場合は -= np.dot(other.vel - person.vel, normal) * normal.
                        other.vel -= (2 * np.dot(other.vel, normal)) * normal
                        indices.remove(l)
                        # 感染状態を必要であれば変化させる.
                        if person.state == State.Infected and other.state == State.Susceptible:
                            other.state = State.Infected
                            other.infection_start = i
                        elif person.state == State.Susceptible and other.state == State.Infected:
                            person.state = State.Infected
                            person.infection_start = i
                        break
            # 時間の index を一つ進める.
            i += 1
            # 総感染者数.
            I = 0
            # 総未感染者数.
            S = 0
            # 総免疫獲得者数.
            R = 0
            for person in self.persons:
                # 各人間に対し, 現在の座標を追加記憶させる.
                person.coords = np.vstack([person.coords, person.co])
                if person.state == State.Infected:
                    I += 1
                elif person.state == State.Susceptible:
                    S += 1
                elif person.state == State.Recovered:
                    R += 1
            # もし総感染者数 I が感染者数の閾値（threshold）を
            # 初めて超えた場合, exceeded_threshold を真とする.
            if threshold < I and not exceeded_threshold:
                exceeded_threshold = True
            # 行動制限を解除する場合（self.comeback が真の場合）,
            # もしまだ行動制限を解除しておらず（comebacked が偽）,
            # 総感染者数 I が感染者数の閾値（threshold）を既に超えており（exceeded_threshold が真）
            # その後総感染者数 I が感染者数の閾値（threshold）を下回った場合,
            # 停止させている人間は全て動かす.
            if I <= threshold and self.comeback and not comebacked and exceeded_threshold:
                for person in self.persons:
                    person.move()
                comebacked = True
            # 総感染者数が 0 つまり感染が収束した場合,
            # 感染収束時の時間の index を subsided に記憶させる.
            if I == 0 and subsided is None:
                subsided = i
            if comebacked:
                print("Simulating SIR %d (I: %d, S: %d, R: %d) (Comebacked)" % (i, I, S, R))
            else:
                print("Simulating SIR %d (I: %d, S: %d, R: %d)" % (i, I, S, R))
            # それぞれの数値計算結果を記憶.
            self.times = np.append(self.times, i * dt)
            self.zeros = np.append(self.zeros, 0)
            self.I = np.append(self.I, I)
            self.S = np.append(self.S, S)
            self.R = np.append(self.R, R)
            # 感染収束時の時間の index が subsided に記憶されており,
            # 時間の index（= i）が subsided とバッファの和になったとき,
            # シミュレーションを終了する.
            if subsided is not None and i == subsided + buff:
                self.end = i
                break

    def animate(
        self
    ):
        """
        SIR のシミュレーション & グラフのアニメーションを作成.
        """
        def simulate(i):
            """
            時間の index が i の時のシミュレーションフレームを描画.

            Parameters
            ----------
            i: int
                時間の index.
            """
            print("Animating simulation %1.2f%% (%d / %d)" % ((i + 1) * 100 / self.end, i + 1, self.end))
            plt.cla()
            # 目盛りの数値 & 目盛りを消去.
            plt.tick_params(labelbottom=False,
                            labelleft=False,
                            labelright=False,
                            labeltop=False,
                            bottom=False,
                            left=False,
                            right=False,
                            top=False)
            ax = fig.add_subplot()
            # 正方形 Box の枠の色は白色で設定.
            ax.spines['top'].set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            # x の範囲は -half ~ half まで.
            ax.set_xlim(-half, half)
            # y の範囲は -half ~ half まで.
            ax.set_ylim(-half, half)
            # アスペクト比は等しく設定.
            ax.set_aspect('equal')
            # 背景色は黒で設定.
            ax.set_facecolor('black')
            # 各人間を, 感染が始まっていない場合には緑色の粒子で,
            # 感染している場合には赤色の粒子で,
            # 免疫獲得している場合には青色の粒子で,
            # 時間の index が i の時の各人間の座標の位置に描画.
            for person in self.persons:
                if i < person.infection_start:
                    # この時は未感染状態.
                    circ = patches.Circle(xy=tuple(person.coords[i]),
                                          radius=radius,
                                          fc='lime')
                elif person.infection_start <= i <= person.infection_start + time_to_recover / dt:
                    # この時は感染状態.
                    circ = patches.Circle(xy=tuple(person.coords[i]),
                                          radius=radius,
                                          fc='red')
                    # オーラを周期的に描画.
                    p_aura = 10 * (i - person.infection_start) * dt / time_to_recover
                    aura_radius = 2.5 * (p_aura - int(p_aura)) * radius
                    aura = patches.Circle(xy=tuple(person.coords[i]),
                                          radius=aura_radius,
                                          ec='red',
                                          fill=False)
                    ax.add_patch(aura)
                else:
                    # この時は免疫獲得状態.
                    circ = patches.Circle(xy=tuple(person.coords[i]),
                                          radius=radius,
                                          fc='blue')
                ax.add_patch(circ)
        simula_name = "simula_" + str(self.number) + "_" + str(self.R0)
        if self.p is not None:
            simula_name += "_p=" + str(self.p)
        if self.comeback:
            simula_name += "_comeback"
        index = 0
        while True:
            path = os.path.join(
                this_dir,
                simula_name + "_" + str(index)
            ) + ".mp4"
            if not os.path.exists(path):
                simula_name += "_" + str(index)
                break
            else:
                index += 1
        # シミュレーションのアニメーションを作成.
        animate(update=simulate,
                end=self.end,
                name=simula_name)
        def draw_graph(i):
            """
            時間の index が i の時のグラフフレームを描画.

            Parameters
            ----------
            i: int
                時間の index.
            """
            print("Animating graph %1.2f%% (%d / %d)" % ((i + 1) * 100 / self.end, i + 1, self.end))
            plt.cla()
            # x 軸, y 軸の目盛りの値 & 目盛りは描画.
            plt.tick_params(labelbottom=True,
                            labelleft=True,
                            labelright=False,
                            labeltop=False,
                            bottom=True,
                            left=True,
                            right=False,
                            top=False)
            ax = fig.add_subplot()
            ax.spines['top'].set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            # x 軸目盛りの色は白で設定.
            ax.tick_params(axis='x', colors='white')
            # y 軸目盛りの色は白で設定.
            ax.tick_params(axis='y', colors='white')
            # アスペクト比は自動設定.
            ax.set_aspect('auto')
            ax.set_facecolor('black')
            # 時間の index が i までの時間の配列,
            # ゼロ配列,
            # 感染者数の推移の配列,
            # 未感染者数の推移の配列,
            # 免疫獲得者数の推移の配列,
            # をスライスで取得.
            times = self.times[:i + 1]
            zeros = self.zeros[:i + 1]
            I = self.I[:i + 1]
            S = self.S[:i + 1]
            R = self.R[:i + 1]
            # それぞれの推移を描画.
            ax.fill_between(times,
                            I + S,
                            I + S + R,
                            facecolor='blue',
                            alpha=1.0,
                            label=str(int(self.R[i])) + "（免疫獲得者）")
            ax.fill_between(times,
                            I,
                            I + S,
                            facecolor='lime',
                            alpha=1.0,
                            label=str(int(self.S[i])) + "（未感染者）")
            ax.fill_between(times,
                            zeros,
                            I,
                            facecolor='red',
                            alpha=1.0,
                            label=str(int(self.I[i])) + "（感染者）")
            ax.legend(loc="upper left", prop={"family": "MS Gothic"})
        graph_name = "graph_" + str(self.number) + "_" + str(self.R0)
        if self.p is not None:
            graph_name += "_p=" + str(self.p)
        if self.comeback:
            graph_name += "_comeback"
        graph_name += "_" + str(index)
        # グラフの推移アニメーションを作成.
        animate(update=draw_graph,
                end=self.end,
                name=graph_name)


def animate(
    update,
    end,
    name
):
    """
    matplotlib.animation.FuncAnimation によるアニメーション作成.
    ffmpeg をインストールしておく必要がある.

    Parameters
    ----------
    update: func
        各フレームを描画する関数を指定.
    end: int
        終了フレーム.
    name: str
        アニメーション保存時の名前.
    """
    anim = animation.FuncAnimation(fig,
                                   update,
                                   frames=end,
                                   interval=100/3)
    anim.save(name + ".mp4",
              writer='ffmpeg',
              dpi=300,
              savefig_kwargs={'facecolor':'black'})


if __name__ == "__main__":
    population = Population(number=500,
                            comeback=True,
                            p=0.8)
    population.simulate_sir()
    population.animate()
