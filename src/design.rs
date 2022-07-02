use rand::{self, Rng};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};

#[allow(dead_code)]
struct RangeModule {
    range: Vec<Vec<i32>>,
}

#[allow(dead_code)]
impl RangeModule {
    fn new() -> Self {
        //println!("new an instance");
        Self { range: Vec::new() }
    }

    fn add_range(&mut self, left: i32, right: i32) {
        //println!("ADD: before: {:?}, target: [{}, {})", self.range, left, right);
        if self.range.len() == 0 {
            self.range.push(vec![left, right]);
            //println!("ADD: after: {:?}", self.range);
            return;
        }
        let (mut l, mut r) = (0i32, self.range.len() as i32 - 1);
        while l <= r {
            let mid = l + (r - l) / 2;
            if self.range[mid as usize][0] > left {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        if r != -1 {
            let mut r = r as usize;
            let mut flag = false;
            if self.range[r][1] >= right {
                //println!("ADD: after: {:?}", self.range);
                return;
            } else if self.range[r][1] < right && self.range[r][1] >= left {
                self.range[r][1] = right;
                flag = true;
            }
            let old = r;
            r += 1;
            let mut rm = vec![];
            let mut right = right;
            while r < self.range.len() && self.range[r][0] <= right {
                if self.range[r][1] <= right {
                    rm.push(r);
                } else {
                    if flag == true {
                        self.range[old][1] = self.range[r][1];
                        rm.push(r);
                    } else {
                        right = self.range[r][1];
                        rm.push(r);
                    }
                    break;
                }
                r += 1;
            }
            for i in rm.iter().rev() {
                self.range.remove(*i);
            }
            if flag == false {
                self.range.push(vec![left, right]);
                self.range.sort_by_key(|x| x[0]);
            }
        } else {
            let mut right = right;
            let mut ll = 0;
            let mut rm = vec![];
            while ll < self.range.len() && self.range[ll][0] <= right {
                if self.range[ll][1] <= right {
                    rm.push(ll as usize);
                } else {
                    rm.push(ll as usize);
                    right = self.range[ll][1];
                    break;
                }
                ll += 1;
            }
            //println!("add, rm: {:?}, before rm, range: {:?}", rm, self.range);
            for i in rm.iter().rev() {
                self.range.remove(*i);
            }
            self.range.push(vec![left, right]);
            self.range.sort_by_key(|x| x[0]);
        }
        //println!("ADD: after: {:?}", self.range);
    }

    fn query_range(&self, left: i32, right: i32) -> bool {
        //println!("QRY: range = {:?}, target: [{}, {})", self.range, left, right);
        let (mut l, mut r) = (0i32, self.range.len() as i32 - 1);
        while l <= r {
            let mid = l + (r - l) / 2;
            if self.range[mid as usize][0] > left {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        if r == -1 {
            return false;
        }
        if self.range[r as usize][0] <= left && right <= self.range[r as usize][1] {
            return true;
        }
        false
    }

    fn remove_range(&mut self, left: i32, right: i32) {
        //println!("DEL: before: {:?}, target: [{}, {})", self.range, left, right);
        let (mut l, mut r) = (0i32, self.range.len() as i32 - 1);
        while l <= r {
            let mid = l + (r - l) / 2;
            if self.range[mid as usize][0] > left {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        let l = r as usize;
        if r != -1 {
            if self.range[l][1] >= right {
                if right < self.range[l][1] {
                    self.range.push(vec![right, self.range[l][1]]);
                }
                self.range[l][1] = left;
                if self.range[l][0] == self.range[l][1] {
                    self.range.remove(l);
                }
                self.range.sort_by_key(|x| x[0]);
                //println!("DEL: after: {:?}", self.range);
                return;
            } else if self.range[l][1] < right && self.range[l][1] >= left {
                self.range[l][1] = left;
            }
            let mut ll = l + 1;
            let mut rm = vec![];
            while ll < self.range.len() && self.range[ll][0] <= right {
                if self.range[ll][1] <= right {
                    rm.push(ll);
                } else {
                    self.range[ll][0] = right;
                }
                ll += 1;
            }
            for i in rm.iter().rev() {
                self.range.remove(*i);
            }
        } else {
            let mut ll = 0;
            let mut rm = vec![];
            while ll < self.range.len() && self.range[ll][0] <= right {
                if self.range[ll][1] <= right {
                    rm.push(ll);
                } else {
                    self.range[ll][0] = right;
                }
                ll += 1;
            }
            for i in rm.iter().rev() {
                self.range.remove(*i);
            }
        }
        //println!("DEL: after: {:?}", self.range);
    }
}

#[derive(Debug)]
pub struct CustomStack {
    s: RefCell<Vec<i32>>,
    num: RefCell<i32>,
    max_size: i32,
    tail: RefCell<i32>,
}

impl CustomStack {
    pub fn new(max_size: i32) -> Self {
        CustomStack {
            s: RefCell::new(vec![0; max_size as usize]),
            num: RefCell::new(0),
            max_size: max_size,
            tail: RefCell::new(-1),
        }
    }

    pub fn push(&self, x: i32) {
        if self.num.borrow().eq(&self.max_size) {
            return;
        }
        *self.num.borrow_mut() += 1;
        *self.tail.borrow_mut() += 1;
        let tail = *self.tail.borrow_mut() as usize;
        let s = &mut *self.s.borrow_mut();
        (*s)[tail] = x;
    }

    pub fn pop(&self) -> i32 {
        if self.num.borrow().eq(&0) {
            return 0;
        }
        *self.num.borrow_mut() -= 1;
        let s = &mut *self.s.borrow_mut();
        let tail = *self.tail.borrow_mut() as usize;
        *self.tail.borrow_mut() -= 1;
        s[tail]
    }

    pub fn increment(&self, k: i32, val: i32) {
        let s = &mut *self.s.borrow_mut();
        let len = *self.num.borrow();
        let len = len.min(k);
        for i in 0..len as usize {
            (*s)[i] += val;
        }
    }
}

pub struct NumArray {
    s: RefCell<Vec<i32>>,
}

#[allow(dead_code)]
impl NumArray {
    pub fn new(nums: Vec<i32>) -> Self {
        let mut nums = nums;
        for i in 0..nums.len() {
            if i == 0 {
                continue;
            }
            nums[i] = nums[i] + nums[i - 1];
        }
        Self {
            s: RefCell::new(nums),
        }
    }

    pub fn sum_range(&self, left: i32, right: i32) -> i32 {
        let s = &mut *self.s.borrow_mut();
        let sum;
        if left == 0 {
            sum = (*s)[right as usize];
        } else {
            sum = (*s)[right as usize] - (*s)[left as usize - 1];
        }
        sum
    }
}

pub struct StreamRank {
    stream: HashMap<i32, i32>,
}

impl StreamRank {
    pub fn new() -> Self {
        Self {
            stream: HashMap::new(),
        }
    }

    pub fn track(&mut self, x: i32) {
        let h = self.stream.entry(x).or_insert(0);
        *h += 1;
    }

    pub fn get_rank_of_number(&self, x: i32) -> i32 {
        let h = &self.stream;
        let mut sum = 0;
        for (k, v) in h.iter() {
            if *k <= x {
                sum += v;
            }
        }
        sum
    }
}

pub struct Bank {
    balance: RefCell<Vec<i64>>,
    number_of_account: usize,
}

impl Bank {
    pub fn new(balance: Vec<i64>) -> Self {
        Self {
            balance: RefCell::new(balance.clone()),
            number_of_account: balance.len(),
        }
    }

    pub fn transfer(&self, account1: i32, account2: i32, money: i64) -> bool {
        println!("transfer: {} -> {}, amount: {}", account1, account2, money);
        let n = self.number_of_account as i32;
        if account1 > n || account2 > n {
            return false;
        }
        let mut bal = self.balance.borrow_mut();
        let remain = bal[account1 as usize - 1];
        println!("acount1 remain: {}", remain);
        if remain < money {
            return false;
        } else {
            bal[account1 as usize - 1] -= money;
            bal[account2 as usize - 1] += money;
        }
        true
    }

    pub fn deposit(&self, account: i32, money: i64) -> bool {
        let n = self.number_of_account as i32;
        if account > n {
            return false;
        }
        let mut bal = self.balance.borrow_mut();
        bal[account as usize - 1] += money;
        true
    }

    pub fn withdraw(&self, account: i32, money: i64) -> bool {
        let n = self.number_of_account as i32;
        if account > n {
            return false;
        }
        let mut bal = self.balance.borrow_mut();
        let t = bal[account as usize - 1];
        if t < money {
            return false;
        } else {
            bal[account as usize - 1] -= money;
        }
        true
    }
}

#[allow(dead_code)]
struct RandomizedSet {
    num: Vec<i32>,
    bt: BTreeMap<i32, usize>,
    total: usize,
}

#[allow(dead_code)]
impl RandomizedSet {
    fn new() -> Self {
        Self {
            num: Vec::new(),
            bt: BTreeMap::new(),
            total: 0,
        }
    }

    fn insert(&mut self, val: i32) -> bool {
        if let Some(_idx) = self.bt.get(&val) {
            return false;
        } else {
            self.num.push(val);
            self.bt.insert(val, self.total);
            self.total += 1;
        }
        true
    }

    fn remove(&mut self, val: i32) -> bool {
        if let Some(&idx) = self.bt.get(&val) {
            self.bt.remove(&val);
            self.total -= 1;
            self.num.remove(idx);
            return true;
        }
        false
    }

    fn get_random(&self) -> i32 {
        let mut rnd = rand::thread_rng();
        let r = rnd.gen_range(0..self.total);
        self.num.get(r).unwrap().clone()
    }
}

#[allow(dead_code)]
struct RecentCounter {
    mq: Vec<i32>,
    cnt: i32,
}

#[allow(dead_code)]
impl RecentCounter {
    fn new() -> Self {
        Self {
            mq: Vec::new(),
            cnt: 0,
        }
    }

    fn ping(&mut self, t: i32) -> i32 {
        if self.cnt == 0 {
            self.mq.push(t);
            self.cnt = 1;
            return 1;
        }
        let mut index = 0;
        if self.mq[0] >= t - 3000 {
            self.mq.push(t);
            self.cnt += 1;
            return self.cnt;
        }
        self.mq.iter().enumerate().for_each(|(i, f)| {
            if *f < t - 3000 {
                index = i;
            }
        });
        self.mq.push(t);
        self.mq = self.mq.drain(index + 1..).collect();
        self.cnt = self.mq.len() as i32;
        return self.cnt;
    }
}

#[allow(dead_code)]
struct MinStack {
    st: Vec<i32>,
    helper: Vec<i32>,
    len: i32,
}

#[allow(dead_code)]
impl MinStack {
    fn new() -> Self {
        Self {
            st: Vec::new(),
            helper: Vec::new(),
            len: -1,
        }
    }

    fn push(&mut self, val: i32) {
        self.st.push(val);
        self.helper.push(self.get_min().min(val));
        self.len += 1;
    }

    fn pop(&mut self) {
        self.st.pop();
        self.helper.pop();
        self.len -= 1;
    }

    fn top(&self) -> i32 {
        self.st[self.len as usize]
    }

    fn get_min(&self) -> i32 {
        if self.len == -1 {
            return i32::MAX;
        }
        self.helper[self.len as usize]
    }
}

#[allow(dead_code)]
struct Solution0 {
    radius: f64,
    x_center: f64,
    y_center: f64,
}

#[allow(dead_code)]
impl Solution0 {
    fn new(radius: f64, x_center: f64, y_center: f64) -> Self {
        Self {
            radius,
            x_center,
            y_center,
        }
    }

    fn rand_point(&self) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        loop {
            let x = rng.gen_range(-self.radius..=self.radius);
            let y = rng.gen_range(-self.radius..=self.radius);
            if x * x + y * y <= self.radius * self.radius {
                return vec![self.x_center + x, self.y_center + y];
            }
        }
    }
}

#[allow(dead_code)]
struct MyCalendarThree {
    bp: BTreeMap<i32, i32>,
}

#[allow(dead_code)]
impl MyCalendarThree {
    fn new() -> Self {
        Self {
            bp: BTreeMap::new(),
        }
    }

    fn book(&mut self, start: i32, end: i32) -> i32 {
        let e = self.bp.entry(start).or_insert(0);
        *e += 1;
        let e = self.bp.entry(end).or_insert(0);
        *e -= 1;
        let mut ans = 0;
        let mut max = 0;
        for (_, v) in self.bp.iter() {
            ans += *v;
            max = max.max(ans);
        }
        max
    }
}

#[allow(dead_code)]
struct MyCalendarTwo {
    bp: BTreeMap<i32, i32>,
}

#[allow(dead_code)]
impl MyCalendarTwo {
    fn new() -> Self {
        Self {
            bp: BTreeMap::new(),
        }
    }

    fn book(&mut self, start: i32, end: i32) -> bool {
        let e = self.bp.entry(start).or_insert(0);
        *e += 1;
        let e = self.bp.entry(end).or_insert(0);
        *e -= 1;
        let mut ans = 0;
        let mut max = 0;
        let mut flag = true;
        for (_, v) in self.bp.iter() {
            ans += *v;
            max = max.max(ans);
            if max >= 3 {
                flag = false;
                break;
            }
        }
        if flag == false {
            let e = self.bp.entry(start).or_insert(0);
            *e -= 1;
            let e = self.bp.entry(end).or_insert(0);
            *e += 1;
        }
        flag
    }
}

pub struct Solution {
    num: i32,
    index: Vec<i32>,
    rets: Vec<Vec<i32>>,
}

impl Solution {
    pub fn new(rects: Vec<Vec<i32>>) -> Self {
        let mut s = Solution {
            num: 0,
            index: Vec::new(),
            rets: rects,
        };
        s.rets.sort_by(|a, b| a[0].cmp(&b[0]));
        for i in 0..s.rets.len() {
            s.index.push(s.num);
            s.num += (s.rets[i][2] - s.rets[i][0] + 1) * (s.rets[i][3] - s.rets[i][1] + 1);
        }
        s
    }

    pub fn pick(&self) -> Vec<i32> {
        let n = self.num;
        let mut r = rand::thread_rng();
        let index = r.gen_range(0..n);
        let (mut l, mut r) = (0, self.index.len() - 1);
        let mut ret_index = 0;
        while l <= r {
            let mid = l + (r - l) / 2;
            if self.index[mid] > index {
                r = mid - 1;
            } else {
                ret_index = mid;
                l = mid + 1;
            }
        }
        let diff = index - self.index[ret_index];
        let y = diff / (self.rets[ret_index][2] - self.rets[ret_index][0] + 1)
            + self.rets[ret_index][1];
        let x = diff % (self.rets[ret_index][2] - self.rets[ret_index][0] + 1)
            + self.rets[ret_index][0];
        vec![x, y]
    }
}

#[allow(dead_code)]
struct Solution710 {
    bound: i32,
    hp: HashMap<i32, i32>,
}

#[allow(dead_code)]
impl Solution710 {
    fn new(n: i32, blacklist: Vec<i32>) -> Self {
        let mut hp = HashMap::new();
        let bound = n - blacklist.len() as i32;
        let mut black = vec![];
        for &b in blacklist.iter() {
            if b >= bound {
                black.push(b);
            }
        }
        let mut w = bound;
        for &b in blacklist.iter() {
            if b < bound {
                while black.contains(&w) {
                    w += 1;
                }
                hp.insert(b, w);
                w += 1;
            }
        }
        Self { bound, hp }
    }

    fn pick(&self) -> i32 {
        let mut rng = rand::thread_rng();
        let n = rng.gen_range(0..self.bound);
        if let Some(x) = self.hp.get(&n) {
            return *x;
        }
        n
    }
}

#[allow(dead_code)]
struct Codec {
    db: HashMap<i32, String>,
    cnt: i32,
}

#[allow(dead_code)]
impl Codec {
    fn new() -> Self {
        Self {
            db: HashMap::new(),
            cnt: 0,
        }
    }

    #[allow(non_snake_case)]
    // Encodes a URL to a shortened URL.
    fn encode(&mut self, longURL: String) -> String {
        self.db.insert(self.cnt + 1, longURL);
        self.cnt += 1;
        self.cnt.to_string()
    }

    #[allow(non_snake_case)]
    // Decodes a shortened URL to its original URL.
    fn decode(&self, shortURL: String) -> String {
        self.db
            .get(&shortURL.parse::<i32>().unwrap())
            .unwrap()
            .to_owned()
    }
}
