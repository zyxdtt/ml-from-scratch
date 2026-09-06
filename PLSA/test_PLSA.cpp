#include <iostream>
#include <iomanip>
#include "PLSA.hpp"

using namespace std;

int main() {
    // ci=5个词, ben=4篇文档, ti=2个主题
    int ci = 5, ben = 4, ti = 2;

    // x[词][文档]：词频矩阵
    // 文档0,1 偏体育（词0,1高频）
    // 文档2,3 偏金融（词2,3高频）
    mat x(ci, point(ben));
    x[0][0] = 8; x[0][1] = 7; x[0][2] = 1; x[0][3] = 1;
    x[1][0] = 6; x[1][1] = 5; x[1][2] = 1; x[1][3] = 1;
    x[2][0] = 1; x[2][1] = 2; x[2][2] = 8; x[2][3] = 9;
    x[3][0] = 1; x[3][1] = 1; x[3][2] = 7; x[3][3] = 6;
    x[4][0] = 0; x[4][1] = 0; x[4][2] = 0; x[4][3] = 0;

    PLSA model(ci, ben, ti);
    auto [citi, tiben] = model.train(x, 200);

    // citi[词][主题] = P(z|w)，大小 ci × ti
    cout << "=== citi[words][topic] P(z|w) ===" << endl;
    for (int w = 0; w < ci; w++) {
        cout << "words" << w << ": ";
        for (int z = 0; z < ti; z++)
            cout << fixed << setprecision(3) << citi[w][z] << " ";
        cout << endl;
    }

    // tiben[主题][文档] = P(d|z)，大小 ti × ben
    cout << "\n=== tiben[topic][para] P(d|z) ===" << endl;
    for (int z = 0; z < ti; z++) {
        cout << "topic" << z << ": ";
        for (int d = 0; d < ben; d++)
            cout << fixed << setprecision(3) << tiben[z][d] << " ";
        cout << endl;
    }

    return 0;
}