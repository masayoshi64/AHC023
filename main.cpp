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
const vector<pair<int, int>> dxy2 = {{1, 1}, {-1, -1}, {-1, 1}, {1, -1}};

int T, H, W, i0, K;
mat<int> h, v, dist;
vi S, D;
double coef_dist = 0.1, empty_penalty = 50, equal_bonus = 5, wall_penalty = 3;

bool _exists_wall(int x, int y, int nx, int ny){
    if(x == nx){
        if(y > ny) swap(y, ny);
        return v[x][y] == 1;
    }else{
        if(x > nx) swap(x, nx);
        return h[x][y] == 1;
    }
}

bool exists_wall(int x, int y, int nx, int ny){
    if(x == nx || y == ny) return _exists_wall(x, y, nx, ny);
    return (_exists_wall(x, y, nx, y) || _exists_wall(nx, y, nx, ny)) && (_exists_wall(x, y, x, ny) || _exists_wall(x, ny, nx, ny));
}

vector<pair<int, int>> get_valid_places(mat<int>& maze, int d, bool init = false){
    set<pair<int, int>> aps;
    vector<pair<int, int>> path;

    Graph G(H * W);
    rep(x, H)rep(y, W){
        if(maze[x][y] == -1){
            path.emplace_back(x, y);
            for(auto [dx, dy] : dxy){
                int nx = x + dx;
                int ny = y + dy;
                if(nx < 0 || nx >= H || ny < 0 || ny >= W || exists_wall(x, y, nx, ny) || maze[nx][ny] != -1) continue;
                G[x * W + y].emplace_back(nx * W + ny);
            }
        }
    }
    LowLink lowlink_path(G);
    for(int x: lowlink_path.aps){
        aps.insert({x / W, x % W});
    }

    if(!init){
        mat<pair<int, int>> d_to_places(T);
        rep(x, H)rep(y, W){
            if(maze[x][y] != -1) d_to_places[D[maze[x][y]]].emplace_back(x, y);
        }
        mat<int> connected(H, vi(W, 0));
        rep(s, d){
            for(auto [x, y]: d_to_places[s]){
                if(connected[x][y] == 1) continue;
                deque<pair<int, int>> que;
                que.emplace_back(x, y);
                connected[x][y] = 1;
                while(!que.empty()){
                    auto [x, y] = que.front();
                    que.pop_front();
                    for(auto [dx, dy] : dxy){
                        int nx = x + dx;
                        int ny = y + dy;
                        if(nx < 0 || nx >= H || ny < 0 || ny >= W || exists_wall(x, y, nx, ny)) continue;
                        if(maze[nx][ny] == -1){
                            G[x * W + y].emplace_back(nx * W + ny);
                            G[nx * W + ny].emplace_back(x * W + y);
                        }else if(D[maze[nx][ny]] == D[maze[x][y]]){
                            if(connected[nx][ny] == 1) continue;
                            G[x * W + y].emplace_back(nx * W + ny);
                            G[nx * W + ny].emplace_back(x * W + y);
                            connected[nx][ny] = 1;
                            que.emplace_back(nx, ny);
                        }else if(d > D[maze[nx][ny]] && D[maze[nx][ny]] > D[maze[x][y]]){
                            if(connected[nx][ny] == 1) continue;
                            connected[nx][ny] = 1;
                            que.emplace_back(nx, ny);
                        }
                    }
                }
            }
        }
        LowLink lowlink(G);
        for(int x: lowlink.aps){
            aps.insert({x / W, x % W});
        }
    }
    
    vector<pair<int, int>> valid_places;

    for(auto [x, y]: path){
        if(aps.count({x, y}) || (x == i0 && y == 0) || maze[x][y] != -1) continue;
        valid_places.emplace_back(x, y);
    }
    return valid_places;
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

double calc_score(int x, int y, int d, mat<int>& maze){
    double score = -coef_dist * dist[x][y];
    for(auto [dx, dy]: dxy){
        int nx = x + dx;
        int ny = y + dy;
        if(nx < 0 || nx >= H || ny < 0 || ny >= W || exists_wall(x, y, nx, ny)){
            score += wall_penalty;
            continue;
        }
        if(maze[nx][ny] == -1) score += empty_penalty;
        else{
            score += abs(D[maze[nx][ny]] - d);
            if(D[maze[nx][ny]] == d) score -= equal_bonus;
        }
    }
    for(auto [dx, dy]: dxy2){
        int nx = x + dx;
        int ny = y + dy;
        if(nx < 0 || nx >= H || ny < 0 || ny >= W || exists_wall(x, y, nx, ny)) continue;
        if(maze[nx][ny] != -1 && D[maze[nx][ny]] == d) score -= equal_bonus;
    }
    return score;
}

mat<int> greedy(int s, vi& crops, mat<int> maze){
    for(int k: crops){
        vector<tuple<double, int, int>> score_xy;
        for(auto [x, y]: get_valid_places(maze, D[k], s == 0)){
            double score = calc_score(x, y, D[k], maze);
            score_xy.emplace_back(score, x, y);
        }
        if(score_xy.empty()) continue;
        auto [score, x, y] = *min_element(all(score_xy));
        maze[x][y] = k;
    }
    return maze;
}

int main(int argc, char *argv[]) {
    cin.tie(0);
    ios::sync_with_stdio(0);
    cout << setprecision(30) << fixed;
    cerr << setprecision(30) << fixed;

    // input
    if (argc == 5){
        coef_dist = atof(argv[1]);
        empty_penalty = atof(argv[2]);
        equal_bonus = atof(argv[3]);
        wall_penalty = atof(argv[4]);
    }
    cin >> T >> H >> W >> i0;
    h.resize(H - 1, vi(W));
    v.resize(H, vi(W - 1));
    int wall_cnt = 0;
    rep(i, H - 1){
        rep(j, W){
            char x;
            cin >> x;
            h[i][j] = x - '0';
            if(h[i][j] == 1) wall_cnt++; 
        }
    }
    rep(i, H){
        rep(j, W - 1){
            char x;
            cin >> x;
            v[i][j] = x - '0';
            if(v[i][j] == 1) wall_cnt++;
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
    vi X, Y, plant_times, is_planted(K);
    X.resize(K), Y.resize(K), plant_times.resize(K, -1);
    mat<int> maze(H, vi(W, -1));
    dist = calc_dist(maze);
    
    // 初期配置
    vi init_crops;
    rep(s, T){
        rep(j, K){
            if(init_crops.size() == W * H - 1) break;
            if(S[j] == s){
                init_crops.pb(j);
            }
        }
    }
    sort(all(init_crops), [&](int i, int j){
        return D[i] > D[j];
    });
    maze = greedy(0, init_crops, maze);
    rep(x, H)rep(y, W){
        if(maze[x][y] != -1){
            X[maze[x][y]] = x;
            Y[maze[x][y]] = y;
            plant_times[maze[x][y]] = 0;
            is_planted[maze[x][y]] = 1;
        }
    }

    // その他の日
    rep(s, 1, T){
        vi crops;
        rep(j, K){
            if(S[j] == s && plant_times[j] == -1){
                crops.pb(j);
            }
        }

        sort(all(crops), [&](int i, int j){
            return D[i] > D[j];
        });
        maze = greedy(s, crops, maze);

        rep(x, H)rep(y, W){
            if(maze[x][y] != -1 && !is_planted[maze[x][y]]){
                X[maze[x][y]] = x;
                Y[maze[x][y]] = y;
                plant_times[maze[x][y]] = s;
                is_planted[maze[x][y]] = 1;
            }
        }
        
        rep(i, H)rep(j, W){
            if(maze[i][j] != -1 && D[maze[i][j]] == s) maze[i][j] = -1;
        }
    }   
    
    // output
    double score = 0;
    vector<tuple<int, int, int, int>> ans;
    vi plant_cnt(T), all_cnt(T);
    rep(k, K){
        all_cnt[S[k]]++;
        if(plant_times[k] != -1){
            ans.emplace_back(k, X[k], Y[k], plant_times[k]);
            plant_cnt[S[k]]++;
        }
    }
    cout << ans.size() << endl;
    for(auto [k, x, y, s] : ans){
        score += D[k] - S[k] + 1;
        cout << k + 1 << " " << x << " " << y << " " << s + 1 << endl;
    }

    // check
    rep(i, T){
        cerr << "day " << i << ": " << plant_cnt[i] << " / " << all_cnt[i] << endl;
    }
    cerr << "all: " << accumulate(all(plant_cnt), 0) << " / " << accumulate(all(all_cnt), 0) << endl;
    score *= (double)1e6 / (H * W * T);
    cerr << "Score = " << score << endl;
    cerr << "[DATA] wall_cnt = " << wall_cnt << endl;
}