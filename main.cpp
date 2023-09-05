/* #region header */

#pragma GCC optimize("Ofast")
#include <bits/stdc++.h>
using namespace std;
// types
using ll = long long;
using ull = unsigned long long;
using ld = long double;
typedef pair<ll, ll> Pl;
typedef vector<ll> vl;
typedef vector<int> vi;
typedef vector<char> vc;
template <typename T> using mat = vector<vector<T>>;
typedef vector<vector<int>> vvi;
typedef vector<vector<long long>> vvl;
typedef vector<vector<char>> vvc;
// abreviations
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()
#define rep_(i , a_, b_, a, b, ...) for (ll i = (a); i < b; i++)
#define rep(i, ...) rep_(i, __VA_ARGS__, __VA_ARGS__, 0, __VA_ARGS__)
#define rrep_(i, a_, b_, a, b, ...) for (ll i = (b - 1); i >= a; i--)
#define rrep(i, ...) rrep_(i, __VA_ARGS__, __VA_ARGS__, 0, __VA_ARGS__)
#define srep(i, a, b, c) for (ll i = (a); i < b; i += c)
#define SZ(x) ((int)(x).size())
#define pb(x) push_back(x)
#define eb(x) emplace_back(x)
#define mp make_pair
//入出力
#define print(x) cout << x << endl
template <class T> ostream &operator<<(ostream &os, const vector<T> &v) {
    for (auto &e : v)
        cout << e << " ";
    cout << endl;
    return os;
}
void scan(int &a) {
    cin >> a;
}
void scan(long long &a) {
    cin >> a;
}
void scan(char &a) {
    cin >> a;
}
void scan(double &a) {
    cin >> a;
}
void scan(string &a) {
    cin >> a;
}
template <class T> void scan(vector<T> &a) {
    for (auto &i : a)
        scan(i);
}
#define vsum(x) accumulate(all(x), 0LL)
#define vmax(a) *max_element(all(a))
#define vmin(a) *min_element(all(a))
#define lb(c, x) distance((c).begin(), lower_bound(all(c), (x)))
#define ub(c, x) distance((c).begin(), upper_bound(all(c), (x)))
// functions
// gcd(0, x) fails.
ll gcd(ll a, ll b) {
    return b ? gcd(b, a % b) : a;
}
ll lcm(ll a, ll b) {
    return a / gcd(a, b) * b;
}
ll safemod(ll a, ll b) {
    return (a % b + b) % b;
}
template <class T> bool chmax(T &a, const T &b) {
    if (a < b) {
        a = b;
        return 1;
    }
    return 0;
}
template <class T> bool chmin(T &a, const T &b) {
    if (b < a) {
        a = b;
        return 1;
    }
    return 0;
}
template <typename T> T mypow(T x, ll n) {
    T ret = 1;
    while (n > 0) {
        if (n & 1)
            (ret *= x);
        (x *= x);
        n >>= 1;
    }
    return ret;
}
ll modpow(ll x, ll n, const ll mod) {
    ll ret = 1;
    while (n > 0) {
        if (n & 1)
            (ret *= x);
        (x *= x);
        n >>= 1;
        x %= mod;
        ret %= mod;
    }
    return ret;
}

uint64_t my_rand(void) {
    static uint64_t x = 88172645463325252ULL;
    x = x ^ (x << 13);
    x = x ^ (x >> 7);
    return x = x ^ (x << 17);
}
int popcnt(ull x) {
    return __builtin_popcountll(x);
}
template <typename T> vector<int> IOTA(vector<T> a) {
    int n = a.size();
    vector<int> id(n);
    iota(all(id), 0);
    sort(all(id), [&](int i, int j) { return a[i] < a[j]; });
    return id;
}
struct Timer {
    clock_t start_time;
    Timer() {
        start_time = clock();
    }
    void reset() {
        start_time = clock();
    }
    int lap() {
        // return x ms.
        return (clock() - start_time) * 1000 / CLOCKS_PER_SEC;
    }
};
template <int Mod> struct modint {
    int x;

    modint() : x(0) {
    }

    modint(long long y) : x(y >= 0 ? y % Mod : (Mod - (-y) % Mod) % Mod) {
    }

    modint &operator+=(const modint &p) {
        if ((x += p.x) >= Mod)
            x -= Mod;
        return *this;
    }

    modint &operator-=(const modint &p) {
        if ((x += Mod - p.x) >= Mod)
            x -= Mod;
        return *this;
    }

    modint &operator*=(const modint &p) {
        x = (int)(1LL * x * p.x % Mod);
        return *this;
    }

    modint &operator/=(const modint &p) {
        *this *= p.inverse();
        return *this;
    }

    modint operator-() const {
        return modint(-x);
    }

    modint operator+(const modint &p) const {
        return modint(*this) += p;
    }

    modint operator-(const modint &p) const {
        return modint(*this) -= p;
    }

    modint operator*(const modint &p) const {
        return modint(*this) *= p;
    }

    modint operator/(const modint &p) const {
        return modint(*this) /= p;
    }

    bool operator==(const modint &p) const {
        return x == p.x;
    }

    bool operator!=(const modint &p) const {
        return x != p.x;
    }

    modint inverse() const {
        int a = x, b = Mod, u = 1, v = 0, t;
        while (b > 0) {
            t = a / b;
            swap(a -= t * b, b);
            swap(u -= t * v, v);
        }
        return modint(u);
    }

    modint pow(int64_t n) const {
        modint ret(1), mul(x);
        while (n > 0) {
            if (n & 1)
                ret *= mul;
            mul *= mul;
            n >>= 1;
        }
        return ret;
    }

    friend ostream &operator<<(ostream &os, const modint &p) {
        return os << p.x;
    }

    friend istream &operator>>(istream &is, modint &a) {
        long long t;
        is >> t;
        a = modint<Mod>(t);
        return (is);
    }

    static int get_mod() {
        return Mod;
    }

    constexpr int get() const {
        return x;
    }
};

/* #endregion*/
// constant
#define inf 1000000000ll
#define INF 4000000004000000000LL

long long xor64(long long range) {
    static uint64_t x = 88172645463325252ULL;
    x ^= x << 13;
    x ^= x >> 7;
    return (x ^= x << 17) % range;
}

bool time_check() {
    static Timer timer;
    if (timer.lap() > 1500)
        return false;
    return true;
}

struct Edge {
    int to;
    Edge(int to): to(to) {}
};
using Graph = vector<vector<Edge>>;
using P = pair<long, long>;

/* Lowlink: グラフの関節点・橋を列挙する構造体
    作成: O(E+V)
    関節点の集合: vector<int> aps
    橋の集合: vector<P> bridges
*/
struct LowLink {
    const Graph &G;
    vector<int> used, ord, low;
    vector<int> aps;  // articulation points
    vector<P> bridges;

    LowLink(const Graph &G_) : G(G_) {
        used.assign(G.size(), 0);
        ord.assign(G.size(), 0);
        low.assign(G.size(), 0);
        int k = 0;
        for (int i = 0; i < (int)G.size(); i++) {
            if (!used[i]) k = dfs(i, k, -1);
        }
        sort(aps.begin(), aps.end()); // 必要ならソートする
        sort(bridges.begin(), bridges.end()); // 必要ならソートする
    }

    int dfs(int id, int k, int par) { // id:探索中の頂点, k:dfsで何番目に探索するか, par:idの親
        used[id] = true;
        ord[id] = k++;
        low[id] = ord[id];
        bool is_aps = false;
        int count = 0; // 子の数
        for (auto &e : G[id]) {
            if (!used[e.to]) {
                count++;
                k = dfs(e.to, k, id);
                low[id] = min(low[id], low[e.to]);
                if (par != -1 && ord[id] <= low[e.to]) is_aps = true; 
                if (ord[id] < low[e.to]) bridges.emplace_back(min(id, e.to), max(id, e.to)); // 条件を満たすので橋  
            } else if (e.to != par) { // eが後退辺の時
                low[id] = min(low[id], ord[e.to]);
            }
        }
        if (par == -1 && count >= 2) is_aps = true; 
        if (is_aps) aps.push_back(id);
        return k;
    }
};

const vector<pair<int, int>> dxy = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

int T, H, W, i0, K;
mat<int> h, v, dist;
vi S, D;

bool exists_wall(int x, int y, int nx, int ny){
    if(x == nx){
        if(y > ny) swap(y, ny);
        return v[x][y] == 1;
    }else{
        if(x > nx) swap(x, nx);
        return h[x][y] == 1;
    }
}

bool check_maze(mat<int> &maze) {
    mat<bool> used(H, vector<bool>(W, 0));
    deque<pair<int, int>> que;
    que.emplace_back(i0, 0);
    used[i0][0] = true;
    int cnt = 0;
    while (!que.empty()) {
        auto [x, y] = que.front();
        que.pop_front();
        cnt++;
        for(auto [dx, dy] : dxy){
            int nx = x + dx;
            int ny = y + dy;
            if(nx < 0 || nx >= H || ny < 0 || ny >= W || exists_wall(x, y, nx, ny)) continue;
            if(used[nx][ny]) continue;
            if(maze[nx][ny] < maze[x][y]) continue;
            used[nx][ny] = true;
            que.emplace_back(nx, ny);
        }
    }
    return cnt == H * W;
}

mat<int> calc_dist(mat<int>& maze){
    mat<int> dist(H, vi(W, inf));
    deque<pair<int, int>> que;
    if(maze[i0][0] == -1){
        dist[i0][0] = 0;
        que.emplace_back(i0, 0);
    }
    while(!que.empty()){
        auto [x, y] = que.front();
        que.pop_front();
        for(auto [dx, dy] : dxy){
            int nx = x + dx;
            int ny = y + dy;
            if(nx < 0 || nx >= H || ny < 0 || ny >= W || exists_wall(x, y, nx, ny)) continue;
            if(dist[nx][ny] <= dist[x][y] + 1) continue;
            if(maze[nx][ny] != -1) continue;
            dist[nx][ny] = dist[x][y] + 1;
            que.emplace_back(nx, ny);
        }
    }
    return dist;
}

void hill_climbing(mat<int>& maze, mat<int>& prev_maze, vi& X, vi& Y, vi& plant_times, int s){
    if(s == 0) return;
    Timer timer;
    double max_time = 10;
    vi plants, plants_prev;
    rep(k, K){
        if(plant_times[k] == s){
            plants.pb(k);
        }
        if(plant_times[k] == s - 1){
            plants_prev.pb(k);
        }
    }
    if(plants.empty() || plants_prev.empty()) return;

    while (timer.lap() < max_time) {
        int k = plants[xor64(plants.size())];
        int pk = plants_prev[xor64(plants_prev.size())];
        int x = X[k], y = Y[k], px = X[pk], py = Y[pk];
        if((dist[x][y] < dist[px][py]) && (maze[x][y] < maze[px][py]))continue;
        if((dist[x][y] > dist[px][py]) && (maze[x][y] > maze[px][py]))continue;
        if(prev_maze[x][y] != -1) continue;
        swap(maze[x][y], maze[px][py]);
        swap(prev_maze[x][y], prev_maze[px][py]);
        prev_maze[px][py] = D[k];
        if(check_maze(maze) && check_maze(prev_maze)){
            swap(X[k], X[pk]);
            swap(Y[k], Y[pk]);
            plant_times[k] = s - 1;
        }else{
            prev_maze[px][py] = -1;
            swap(prev_maze[x][y], prev_maze[px][py]);
            swap(maze[x][y], maze[px][py]);
        }
    }
}

int main(int argc, char *argv[]) {
    cin.tie(0);
    ios::sync_with_stdio(0);
    cout << setprecision(30) << fixed;
    cerr << setprecision(30) << fixed;

    // input
    cin >> T >> H >> W >> i0;
    h.resize(H - 1, vi(W));
    v.resize(H, vi(W - 1));
    rep(i, H - 1){
        rep(j, W){
            char x;
            cin >> x;
            h[i][j] = x - '0';
        }
    }
    rep(i, H){
        rep(j, W - 1){
            char x;
            cin >> x;
            v[i][j] = x - '0';
        }
    }
    cin >> K;
    S.resize(K);
    D.resize(K);
    rep(i, K) {
        cin >> S[i] >> D[i];
        S[i]--;
        D[i]--;
    }

    // solve
    vi X(K), Y(K), plant_times(K, -1);
    vector<bool> used(K, false);
    mat<int> maze(H, vi(W, -1)), prev_maze(H, vi(W, -1));
    dist = calc_dist(maze);
    rep(s, T){
        vi crops;
        rep(j, K){
            if(S[j] == s && !used[j]){
                crops.pb(j);
            }
        }
        if(crops.empty()) continue;
        sort(all(crops), [&](int i, int j){
            return D[i] > D[j];
        });

        for(int k: crops){
            Graph G(H * W);
            rep(x, H)rep(y, W){
                if(maze[x][y] == -1){
                    for(auto [dx, dy] : dxy){
                        int nx = x + dx;
                        int ny = y + dy;
                        if(nx < 0 || nx >= H || ny < 0 || ny >= W || exists_wall(x, y, nx, ny) || maze[nx][ny] != -1) continue;
                        G[x * W + y].emplace_back(nx * W + ny);
                    }
                }
            }
            LowLink lowlink(G);
            set<pair<int, int>> aps;
            for(int x: lowlink.aps){
                aps.insert({x / W, x % W});
            }
            vector<tuple<double, int, int>> score_xy;
            rep(x, H)rep(y, W){
                if(maze[x][y] != -1 || aps.count({x, y})) continue;
                double score = -(double)dist[x][y] + 0.001 * (min(x, H-x) + min(y, W-y));
                score_xy.emplace_back(score, x, y);
            }
            sort(all(score_xy));
            for(auto [d, x, y]: score_xy){
                maze[x][y] = D[k];
                if(check_maze(maze)){
                    used[k] = true;
                    X[k] = x, Y[k] = y, plant_times[k] = s;
                    break;
                }else{
                    maze[x][y] = -1;
                }
            }
        }
        // hill_climbing(maze, prev_maze, X, Y, plant_times, s);
        prev_maze = maze;
        rep(i, H)rep(j, W){
            if(maze[i][j] == s)maze[i][j] = -1;
        }
    }   
    
    // output
    double score = 0;
    vector<tuple<int, int, int, int>> ans;
    rep(k, K){
        if(plant_times[k] != -1){
            ans.emplace_back(k, X[k], Y[k], plant_times[k]);
        }
    }
    cout << ans.size() << endl;
    for(auto [k, x, y, s] : ans){
        score += D[k] - S[k] + 1;
        cout << k + 1 << " " << x << " " << y << " " << s + 1 << endl;
    }
    score *= (double)1e6 / (H * W * T);
    cerr << "Score = " << score << endl;
}