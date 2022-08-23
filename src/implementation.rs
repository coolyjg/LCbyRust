use rand::{self, random, Rng};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::vec;

pub fn minimum_difference(nums: Vec<i32>, k: i32) -> i32 {
    if k == 1 {
        return 0;
    }
    let mut nums = nums;
    let k = k as usize;
    nums.sort_unstable();
    *nums
        .windows(k)
        .map(|a| a[k - 1] - a[0])
        .collect::<Vec<i32>>()
        .iter()
        .min()
        .unwrap()
}

pub fn max_number_of_balloons(text: String) -> i32 {
    let mut hp = HashMap::<char, i32>::new();
    for c in text.chars() {
        let e = hp.entry(c).or_insert(0);
        *e += 1;
    }
    let mut ans = i32::MAX;
    if let Some(a) = hp.get(&'a') {
        ans = ans.min(*a);
        if let Some(b) = hp.get(&'b') {
            ans = ans.min(*b);
            if let Some(l) = hp.get(&'l') {
                ans = ans.min(*l / 2);
                if let Some(o) = hp.get(&'o') {
                    ans = ans.min(*o / 2);
                    if let Some(n) = hp.get(&'n') {
                        ans = ans.min(*n);
                    } else {
                        return 0;
                    }
                } else {
                    return 0;
                }
            } else {
                return 0;
            }
        } else {
            return 0;
        }
    } else {
        return 0;
    }
    ans
}

pub fn reverse_parentheses(s: String) -> String {
    let n = s.len();
    let mut s: Vec<char> = s.chars().collect();
    let mut stack: Vec<usize> = vec![];
    for i in 0..n {
        if s[i] == '(' {
            stack.push(i);
        } else if s[i] == ')' {
            let j = stack.pop().unwrap();
            for k in j..((i + j) / 2 + 1) {
                s.swap(k, i + j - k);
            }
        }
    }
    let mut ans = String::new();
    let _ss: Vec<_> = s
        .iter()
        .map(|&a| {
            if a != '(' && a != ')' {
                ans.push(a);
            }
        })
        .collect();
    ans
}

pub fn reverse_str(s: String, k: i32) -> String {
    let mut s = s.chars().collect::<Vec<char>>();
    for i in (0..s.len()).step_by(2 * k as usize) {
        let len = if s.len() - i > k as usize {
            k as usize
        } else {
            s.len() - i
        };
        for j in 0..len / 2 {
            s.swap(i + j, i + len - j - 1);
        }
    }
    s.iter().collect::<String>()
}

pub fn update_matrix(mat: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut ans = mat.clone();
    let mut q = VecDeque::new();
    let mut visited = vec![vec![false; mat[0].len()]; mat.len()];
    for i in 0..mat.len() {
        for j in 0..mat[0].len() {
            if mat[i][j] == 0 {
                q.push_back((i, j));
                visited[i][j] = true;
            }
        }
    }
    while !q.is_empty() {
        for _ in 0..q.len() {
            if let Some((i, j)) = q.pop_front() {
                let (i, j) = (i as i32, j as i32);
                for (r, c) in [(i, j + 1), (i, j - 1), (i + 1, j), (i - 1, j)] {
                    if r >= 0
                        && c >= 0
                        && r < mat.len() as i32
                        && c < mat[0].len() as i32
                        && visited[r as usize][c as usize] == false
                    {
                        ans[r as usize][c as usize] = ans[i as usize][j as usize] + 1;
                        visited[r as usize][c as usize] = true;
                        q.push_back((r as usize, c as usize));
                    }
                }
            }
        }
    }
    ans
}

pub fn lucky_numbers(matrix: Vec<Vec<i32>>) -> Vec<i32> {
    fn check(matrix: &Vec<Vec<i32>>, _i: usize, j: usize, a: i32) -> bool {
        for k in 0..matrix.len() {
            if matrix[k][j] > a {
                return false;
            }
        }
        true
    }
    let mut temp = vec![(i32::MAX, 0, 0); matrix.len()];
    for i in 0..matrix.len() {
        for j in 0..matrix[0].len() {
            if matrix[i][j] < temp[i].0 {
                temp[i].0 = matrix[i][j];
                temp[i].1 = i;
                temp[i].2 = j;
            }
        }
    }
    let mut ans = vec![];
    for num in temp {
        if check(&matrix, num.1, num.2, num.0) {
            ans.push(num.0);
            if ans.len() > 1 {
                return vec![];
            }
        }
    }
    ans
}

pub fn next_permutation(nums: &mut Vec<i32>) {
    let len = nums.len();
    let mut flag = false;
    let mut j = len;
    for i in (0..len).rev() {
        if i == 0 {
            flag = true;
            break;
        }
        if nums[i - 1] < nums[i] {
            j = i - 1;
            break;
        }
    }
    if flag == true {
        reverse(nums, 0, len);
        return;
    }
    for i in (j + 1..len).rev() {
        if nums[i] > nums[j] {
            nums.swap(i, j);
            break;
        }
    }
    reverse(nums, j + 1, len);
    fn reverse(nums: &mut Vec<i32>, a: usize, len: usize) {
        for i in a..(a + (len - a) / 2) {
            nums.swap(i, len - i + a - 1);
        }
    }
}

pub fn convert(s: String, num_rows: i32) -> String {
    let num_rows = num_rows as usize;
    let mut ans = vec![String::new(); num_rows];
    let iter = (0..num_rows).chain((1..num_rows - 1).rev()).cycle();
    iter.zip(s.chars()).for_each(|(i, c)| {
        ans[i].push(c);
    });
    ans.into_iter().collect()
}

pub fn letter_combinations(digits: String) -> Vec<String> {
    let mut ans: Vec<String> = vec![];
    let mut hp = HashMap::new();
    hp.insert('2', "abc".to_string());
    hp.insert('3', "def".to_string());
    hp.insert('4', "ghi".to_string());
    hp.insert('5', "jkl".to_string());
    hp.insert('6', "mno".to_string());
    hp.insert('7', "pqrs".to_string());
    hp.insert('8', "tuv".to_string());
    hp.insert('9', "wxyz".to_string());
    fn dfs(
        temp: &mut String,
        map: &HashMap<char, String>,
        ans: &mut Vec<String>,
        idx: usize,
        digits: &String,
    ) {
        if idx == digits.len() {
            ans.push(temp.clone());
            return;
        }
        let c = digits.chars().nth(idx).unwrap();
        for cc in map.get(&c).unwrap().chars() {
            temp.push(cc);
            dfs(temp, map, ans, idx + 1, digits);
            temp.pop();
        }
    }
    if digits.is_empty() {
        return vec![];
    }
    dfs(&mut String::new(), &hp, &mut ans, 0, &digits);
    ans
}

pub fn knight_probability(n: i32, k: i32, row: i32, column: i32) -> f64 {
    let mut map = vec![vec![vec![0f64; n as usize]; n as usize]; k as usize + 1];
    let m = vec![
        (1, 2),
        (1, -2),
        (-1, 2),
        (-1, -2),
        (2, 1),
        (2, -1),
        (-2, 1),
        (-2, -1),
    ];
    for step in 0..k + 1 {
        for i in 0..n {
            for j in 0..n {
                if step == 0 {
                    map[0][i as usize][j as usize] = 1.0;
                } else {
                    for (ni, nj) in m.iter() {
                        let ii = ni + i;
                        let jj = nj + j;
                        if ii >= 0 && jj >= 0 && ii < n && jj < n {
                            map[step as usize][i as usize][j as usize] +=
                                map[step as usize - 1][ii as usize][jj as usize] / 8.0;
                        }
                    }
                }
            }
        }
    }
    let (k, row, column) = (k as usize, row as usize, column as usize);
    map[k][row][column]
}

pub fn find_center(edges: Vec<Vec<i32>>) -> i32 {
    let a = edges[0][0];
    let b = edges[0][1];
    let c = edges[1][0];
    let d = edges[1][1];
    if c == a || c == b {
        return c;
    } else {
        return d;
    }
}

pub fn compress(chars: &mut Vec<char>) -> i32 {
    let mut l = 0;
    let mut r = 0;
    while r < chars.len() {
        let i = r;
        while r < chars.len() && chars[r] == chars[i] {
            r += 1;
        }
        chars[l] = chars[i];
        l += 1;
        if r > i + 1 {
            let num = format!("{}", r - i);
            for c in num.chars() {
                chars[l] = c;
                l += 1;
            }
        }
    }
    l as i32
}

pub fn minimum_cost(cost: Vec<i32>) -> i32 {
    let mut cost = cost;
    cost.sort_unstable();
    let mut ans = 0;
    let mut cnt = 2;
    for i in (0..cost.len()).rev() {
        if cnt != 0 {
            ans += cost[i];
            cnt -= 1;
        } else {
            cnt = 2;
        }
    }
    ans
}

pub fn pancake_sort(arr: Vec<i32>) -> Vec<i32> {
    let mut i = arr.len();
    let mut ans = vec![];
    let mut arr = arr;
    while i > 0 {
        let (k, _) = arr[0..i].iter().enumerate().max_by_key(|&x| x.1).unwrap();
        if k + 1 != i {
            ans.push(k as i32 + 1);
            arr[0..=k].reverse();
            ans.push(i as i32);
            arr[0..i].reverse();
        }
        i -= 1;
    }
    ans
}

#[allow(unused_assignments)]
pub fn push_dominoes(dominoes: String) -> String {
    let n = dominoes.len();
    let mut dominoes = dominoes.chars().collect::<Vec<char>>();
    let mut i = 0;
    let mut left = 'L';
    let mut right;
    loop {
        if i >= n {
            break;
        }
        let mut j = i;
        loop {
            if j >= n || dominoes[j] != '.' {
                break;
            }
            j += 1;
        }
        if j < n {
            right = dominoes[j];
        } else {
            right = 'R';
        }
        if left == right {
            for k in i..j {
                dominoes[k] = left;
            }
            i = j;
        } else if left == 'R' && right == 'L' {
            let mut k = j - 1;
            loop {
                if i >= k {
                    break;
                }
                dominoes[i] = 'R';
                dominoes[k] = 'L';
                i += 1;
                k -= 1;
            }
        }
        left = right;
        i = j + 1;
    }
    dominoes.iter().collect()
}

pub fn reverse_only_letters(s: String) -> String {
    let mut ss: Vec<char> = s.chars().collect();
    let (mut i, mut j) = (0, ss.len() - 1);
    loop {
        if i >= j {
            break;
        }
        loop {
            if !ss[i].is_ascii_alphabetic() && i < j {
                i += 1;
            } else {
                break;
            }
        }
        loop {
            if !ss[j].is_ascii_alphabetic() && j > i {
                j -= 1;
            } else {
                break;
            }
        }
        if i >= j {
            break;
        }
        ss.swap(i, j);
        i += 1;
        j -= 1;
    }
    ss.iter().collect()
}

pub fn complex_number_multiply(num1: String, num2: String) -> String {
    fn into_number(num: Vec<char>) -> (i32, i32) {
        let (mut r1, mut i1) = (0, 0);
        let mut ii = 0;
        for i in 0..num.len() {
            if num[i] == '-' {
                continue;
            }
            if num[i] != '+' {
                r1 = r1 * 10 + num[i].to_string().parse::<i32>().unwrap();
            } else {
                ii = i;
                break;
            }
        }
        if num[0] == '-' {
            r1 = r1 * (-1);
        }
        for j in ii + 1..num.len() - 1 {
            if num[j] == '-' {
                continue;
            }
            i1 = i1 * 10 + num[j].to_string().parse::<i32>().unwrap();
        }
        if num[ii + 1] == '-' {
            i1 = i1 * (-1);
        }
        (r1, i1)
    }
    let num1 = num1.chars().collect::<Vec<char>>();
    let num2 = num2.chars().collect::<Vec<char>>();
    let (i1, r1) = into_number(num1);
    let (i2, r2) = into_number(num2);
    let i = i1 * i2 - r1 * r2;
    let j = i1 * r2 + i2 * r1;
    let ans = format!("{}+{}i", i, j);
    ans
}

pub fn compare_string(a: String, b: String) -> i32 {
    if a != b {
        return a.len().max(b.len()) as i32;
    }
    -1
}

pub fn good_days_to_rob_bank(security: Vec<i32>, time: i32) -> Vec<i32> {
    if security.len() <= time as usize * 2 {
        return vec![];
    }
    let mut left = vec![0; security.len()];
    let mut right = vec![0; security.len()];
    for i in 1..security.len() {
        if security[i] <= security[i - 1] {
            left[i] = left[i - 1] + 1;
        }
        if security[security.len() - i - 1] <= security[security.len() - i] {
            right[security.len() - i - 1] = right[security.len() - i] + 1;
        }
    }
    let mut ans = vec![];
    for i in time as usize..security.len() - time as usize {
        if left[i] >= time && right[i] >= time {
            ans.push(i as i32);
        }
    }
    ans
}

pub fn convert_to_base7(num: i32) -> String {
    let flag = if num < 0 { true } else { false };
    let mut num = num.abs();
    let mut ans = String::from("");
    loop {
        if num == 0 {
            break;
        }
        let c = num % 7;
        ans.push_str(&c.to_string());
        num = num / 7;
    }
    let ans: String = ans.chars().rev().collect();
    if flag == true {
        return format!("-{}", ans);
    }
    ans
}

pub fn replace_non_coprimes(nums: Vec<i32>) -> Vec<i32> {
    fn gcd(a: i32, b: i32) -> i32 {
        if b == 0 {
            return a;
        } else {
            return gcd(b, a % b);
        }
    }
    let mut ans = vec![nums[0]];
    for &num in nums.iter().skip(1) {
        ans.push(num);
        loop {
            if ans.len() <= 1 {
                break;
            }
            let x = ans[ans.len() - 1];
            let y = ans[ans.len() - 2];
            let g = gcd(x, y);
            if g == 1 {
                break;
            }
            ans.pop();
            let top = ans.last_mut().unwrap();
            *top = *top / g * x;
        }
    }
    ans
}

pub fn is_match(s: String, p: String) -> bool {
    let (m, n) = (s.len(), p.len());
    let mut f = vec![vec![true; n + 1]; m + 1];
    let s = s.chars().collect::<Vec<char>>();
    let p = p.chars().collect::<Vec<char>>();
    let matches_c = |i: usize, j: usize| -> bool {
        if i == 0 {
            false
        } else {
            p[j - 1] == '.' || s[i - 1] == p[j - 1]
        }
    };
    f[0][0] = true;
    for i in 0..=m {
        for j in 1..=n {
            if p[j - 1] == '*' {
                if matches_c(i, j - 1) {
                    f[i][j] = f[i - 1][j] | f[i][j - 2];
                } else {
                    f[i][j] = f[i][j - 2];
                }
            } else {
                f[i][j] = if matches_c(i, j) {
                    f[i - 1][j - 1]
                } else {
                    false
                };
            }
        }
    }
    f[m][n]
}

pub fn plates_between_candles(s: String, queries: Vec<Vec<i32>>) -> Vec<i32> {
    let mut num_of_plate = vec![0; s.len()];
    let mut left = vec![-1; s.len()];
    let mut right = vec![-1; s.len()];
    let s = s.chars().collect::<Vec<char>>();
    s.iter().enumerate().fold((-1, 0), |acc, (i, x)| {
        if *x == '|' {
            left[i] = i as i32;
            num_of_plate[i] = acc.1;
            (i as i32, acc.1)
        } else {
            left[i] = acc.0;
            num_of_plate[i] = acc.1 + 1;
            (acc.0, acc.1 + 1)
        }
    });
    s.iter()
        .rev()
        .enumerate()
        .fold(s.len() as i32, |acc, (i, x)| {
            if *x == '|' {
                right[i] = s.len() as i32 - 1 - i as i32;
                s.len() as i32 - 1 - i as i32
            } else {
                right[i] = acc;
                acc
            }
        });
    let right = right.into_iter().rev().collect::<Vec<i32>>();
    println!(
        "letf:\n{:?}\nright:\n{:?}\nnumber:\n{:?}\n",
        left, right, num_of_plate
    );
    let mut ans = vec![0; queries.len()];
    for (i, query) in queries.iter().enumerate() {
        let (l, r) = (query[0] as usize, query[1] as usize);
        let (ll, rr) = (right[l] as usize, left[r]);
        if ll as i32 >= rr {
            ans[i] = 0;
        } else {
            if rr <= -1 || rr >= s.len() as i32 {
                ans[i] = 0;
            } else {
                ans[i] = num_of_plate[rr as usize] - num_of_plate[ll];
            }
        }
    }
    ans
}

pub fn find_k_distant_indices(nums: Vec<i32>, key: i32, k: i32) -> Vec<i32> {
    let mut index = vec![];
    let mut ans = vec![];
    let len = nums.len() as i32;
    nums.iter().enumerate().for_each(|x| {
        if *x.1 == key {
            index.push(x.0);
        }
    });
    let mut t = -1;
    for i in index {
        let mut start = 0.max(i as i32 - k).max(t) as usize;
        if start as i32 == t {
            start += 1;
        }
        let end = (i as i32 + k).min(len - 1) as usize;
        println!("i = {}, start = {}, end = {}", i, start, end);
        if start > end {
            continue;
        }
        for j in start..=end {
            t = j as i32;
            ans.push(j as i32);
        }
    }
    ans
}

pub fn find_restaurant(list1: Vec<String>, list2: Vec<String>) -> Vec<String> {
    let mut hp = HashMap::new();
    list1.into_iter().enumerate().for_each(|(i, s)| {
        hp.insert(s, i);
    });
    let mut ans = vec![];
    let mut min = 2001;
    for (i, s) in list2.iter().enumerate() {
        if let Some(j) = hp.get(s) {
            if i + j < min {
                min = i + j;
                ans.drain(..);
                ans.push(s.clone());
            } else if i + j == min {
                ans.push(s.clone());
            }
        }
    }
    ans
}

pub fn count_max_or_subsets_2044(nums: Vec<i32>) -> i32 {
    let mut ans = 0;
    let t = 0;
    let mut max = 0;
    fn helper(nums: &Vec<i32>, max: &mut i32, t: i32, ans: &mut i32, pos: usize) {
        if pos == nums.len() {
            if t > *max {
                *max = t;
                *ans = 1;
            } else if t == *max {
                *ans += 1;
            }
            return;
        }
        helper(nums, max, t | nums[pos], ans, pos + 1);
        helper(nums, max, t, ans, pos + 1);
    }
    helper(&nums, &mut max, t, &mut ans, 0);
    ans
}

pub fn divide_array(nums: Vec<i32>) -> bool {
    let mut hp: HashMap<i32, i32> = HashMap::new();
    nums.iter().for_each(|x| {
        let h = hp.entry(*x).or_insert(0);
        *h += 1;
    });
    for (_, v) in hp.iter() {
        if v % 2 != 0 {
            return false;
        }
    }
    true
}

pub fn count_hill_valley(nums: Vec<i32>) -> i32 {
    let mut up = true;
    if nums.len() <= 2 {
        return 0;
    }
    let mut ans = 0;
    let init = nums[0];
    let mut i = 1;
    for _ in 1..nums.len() {
        if nums[i] == init {
            i += 1;
            continue;
        } else {
            break;
        }
    }
    if i == nums.len() {
        return 0;
    }
    if nums[i] < nums[0] {
        up = false;
    }
    i += 1;
    for _ in i..nums.len() {
        if nums[i] == nums[i - 1]
            || nums[i] < nums[i - 1] && up == false
            || nums[i] > nums[i - 1] && up == true
        {
            i += 1;
            continue;
        } else if nums[i] > nums[i - 1] && up == false {
            ans += 1;
            i += 1;
            up = true;
            continue;
        } else if nums[i] < nums[i - 1] && up == true {
            ans += 1;
            i += 1;
            up = false;
            continue;
        }
    }
    ans
}

pub fn count_collisions(directions: String) -> i32 {
    let mut dir = directions.chars().collect::<Vec<char>>();
    let mut ans = 0;
    let mut i = 0;
    loop {
        if i > dir.len() {
            break;
        }
        if dir[i] == 'S' {
            i += 1;
            continue;
        } else if dir[i] == 'L' {
            let mut j = i;
            if i != 0 && dir[i - 1] != 'L' {
                for _ in i..dir.len() {
                    if dir[j] == 'L' {
                        ans += 1;
                        dir[j] = 'S';
                        j += 1;
                    } else {
                        break;
                    }
                }
            }
            i = j
        } else if dir[i] == 'R' {
            let mut j = i;
            loop {
                if j >= dir.len() {
                    break;
                }
                if dir[j] == 'R' {
                    j += 1;
                } else {
                    break;
                }
            }
            if j == dir.len() {
                break;
            } else {
                for k in i..j {
                    dir[k] = 'S';
                    ans += 1;
                }
            }
            i = j;
        }
    }
    ans
}

pub fn reverse_vowels(s: String) -> String {
    let mut ss = s.chars().collect::<Vec<char>>();
    let v = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'];
    let (mut i, mut j) = (0, ss.len() - 1);
    loop {
        if i >= j {
            break;
        }
        loop {
            if i >= j || v.contains(&ss[i]) {
                break;
            }
            i += 1;
        }
        if i >= j {
            break;
        }
        loop {
            if j <= i || v.contains(&ss[j]) {
                break;
            }
            j -= 1;
        }
        if j <= i {
            break;
        }
        ss.swap(i, j);
        i += 1;
        j -= 1;
    }
    ss.iter().collect()
}

pub fn winner_of_game_2038(colors: String) -> bool {
    let color = colors.chars().collect::<Vec<char>>();
    let mut hp = HashMap::from([('A', 0), ('B', 0)]);
    let init = if color[0] == 'A' { ('A', 1) } else { ('B', 1) };
    color.iter().skip(1).fold(init, |next, x| {
        if *x == next.0 && next.1 >= 2 {
            let e = hp.entry(*x).or_insert(0);
            *e += 1;
            (next.0, next.1 + 1)
        } else if *x != next.0 {
            (*x, 1)
        } else {
            (next.0, next.1 + 1)
        }
    });
    let (&aa, &bb) = (hp.get(&'A').unwrap(), hp.get(&'B').unwrap());
    if aa <= bb {
        return false;
    }
    true
}

pub fn missing_rolls(rolls: Vec<i32>, mean: i32, n: i32) -> Vec<i32> {
    let (m, n) = (rolls.len(), n as usize);
    let mut ans = vec![0; n];
    let total = mean * (m as i32 + n as i32);
    let n_mean = (total - rolls.iter().sum::<i32>()) / n as i32;
    let mut n_left = (total - rolls.iter().sum::<i32>()) % n as i32;
    if n_mean > 6 || (n_mean == 6 && n_left != 0) || n_mean <= 0 {
        return vec![];
    }
    ans = ans
        .iter()
        .map(|_| {
            if n_left != 0 {
                n_left -= 1;
                n_mean + 1
            } else {
                n_mean
            }
        })
        .collect();
    ans
}

pub fn has_alternating_bits(n: i32) -> bool {
    let a = n ^ n >> 1;
    return a & (a + 1) == 0;
}

pub fn cal_points(ops: Vec<String>) -> i32 {
    let mut ans = 0;
    let mut score: Vec<i32> = vec![];
    ops.iter().for_each(|s| match s.as_str() {
        "C" => {
            ans -= score.last().unwrap();
            score.pop();
        }
        "D" => {
            let n = score.last().unwrap().to_owned();
            score.push(n * 2);
            ans += score.last().unwrap();
        }
        "+" => {
            let n = score.last().unwrap().to_owned();
            let n2 = score[score.len() - 2];
            score.push(n + n2);
            ans += score.last().unwrap();
        }
        n @ _ => {
            score.push(n.parse::<i32>().unwrap());
            ans += score.last().unwrap();
        }
    });
    ans
}

pub fn max_consecutive_answers(answer_key: String, k: i32) -> i32 {
    let ak = answer_key.chars().collect::<Vec<char>>();
    let ans;
    fn slide_slice(ak: &Vec<char>, k: i32, c: char) -> i32 {
        let (mut i, mut j) = (0, 0);
        let mut max = 0;
        let mut cnt = 0;
        loop {
            if j >= ak.len() {
                break;
            }
            while j < ak.len() && cnt <= k {
                if ak[j] != c {
                    cnt += 1;
                }
                j += 1;
            }
            if j == ak.len() && cnt <= k {
                max = max.max((j - i) as i32);
                return max;
            }
            max = max.max((j - i - 1) as i32);
            while i <= j && ak[i] == c {
                i += 1;
            }
            i += 1;
            cnt -= 1;
        }
        max
    }
    ans = slide_slice(&ak, k, 'T').max(slide_slice(&ak, k, 'F'));
    ans
}

pub fn longest_ones_1004(nums: Vec<i32>, k: i32) -> i32 {
    let mut ans = 0;
    let mut cnt = 0;
    let (mut i, mut j) = (0, 0);
    while j < nums.len() {
        while j < nums.len() && cnt <= k {
            if nums[j] == 0 {
                cnt += 1;
            }
            j += 1;
        }
        if j == nums.len() && cnt <= k {
            return ans.max((j - i) as i32);
        }
        ans = ans.max((j - i - 1) as i32);
        while i <= j && nums[i] == 1 {
            i += 1;
        }
        i += 1;
        cnt -= 1;
    }
    ans
}

pub fn self_dividing_numbers_728(left: i32, right: i32) -> Vec<i32> {
    fn is_self_devided(n: i32) -> bool {
        let mut t = n;
        while t != 0 {
            let c = t % 10;
            if c == 0 || n % c != 0 {
                return false;
            }
            t = t / 10;
        }
        true
    }
    (left..=right)
        .into_iter()
        .filter(|&x| is_self_devided(x))
        .collect()
}

pub fn count_prime_set_bits(left: i32, right: i32) -> i32 {
    (left..=right).fold(0, |acc, i| {
        acc + if 1 << i.count_ones() & 665772 > 0 {
            1
        } else {
            0
        }
    })
}

pub fn rotate_string(s: String, goal: String) -> bool {
    if s.len() != goal.len() {
        return false;
    }
    let ss = s.clone() + &s;
    if let Some(_) = ss.find(&goal) {
        return true;
    }
    false
}

pub fn max_rotate_function(nums: Vec<i32>) -> i32 {
    if nums.len() == 1 {
        return 0;
    }
    let len = nums.len();
    let sum = nums.iter().sum::<i32>();
    let mut value = nums
        .iter()
        .enumerate()
        .fold(0, |acc, (i, x)| acc + i as i32 * x);
    let mut max = value;
    for k in 1..len {
        let temp = value + sum - len as i32 * nums[len - k];
        max = max.max(temp);
        value = temp;
    }
    max
}

pub fn binary_gap(n: i32) -> i32 {
    let mut cnt = 0;
    let mut idx = -1;
    let mut i = -1;
    let mut n = n;
    while n != 0 {
        i += 1;
        let t = n % 2;
        if t == 1 && idx != -1 {
            cnt = cnt.max(i - idx);
        }
        n = n >> 1;
        idx = if t == 1 { i } else { idx };
    }
    cnt
}

pub fn num_subarray_product_less_than_k(nums: Vec<i32>, k: i32) -> i32 {
    let (mut i, mut j) = (0, 0);
    let mut p = 1;
    let mut ret = 0;
    while j < nums.len() {
        p *= nums[j];
        while i <= j && p >= k {
            p /= nums[i];
            i += 1;
        }
        ret += j - i + 1;
        j += 1;
    }
    ret as i32
}

pub fn find_duplicates(nums: Vec<i32>) -> Vec<i32> {
    let mut nums = nums;
    let mut ans = Vec::new();
    for i in 0..nums.len() {
        while nums[nums[i] as usize - 1] != nums[i] {
            let t = nums[i];
            nums[i] = nums[t as usize - 1];
            nums[t as usize - 1] = t;
        }
    }
    nums.iter().enumerate().for_each(|x| {
        if x.0 as i32 != x.1 - 1 {
            ans.push(*x.1);
        }
    });
    ans
}

pub fn min_deletion_size(strs: Vec<String>) -> i32 {
    let strs = strs
        .into_iter()
        .map(|s| s.chars().collect::<Vec<char>>())
        .collect::<Vec<Vec<char>>>();
    let mut ans = 0;
    let (n, m) = (strs.len(), strs[0].len());
    for j in 0..m {
        for i in 1..n {
            if strs[i][j] < strs[i - 1][j] {
                ans += 1;
                break;
            }
        }
    }
    ans
}

pub fn di_string_match(s: String) -> Vec<i32> {
    let mut ans = vec![];
    let (mut low, mut hi) = (0, s.len() as i32);
    s.chars().for_each(|c| {
        if c == 'I' {
            ans.push(low);
            low += 1;
        } else {
            ans.push(hi);
            hi -= 1;
        }
    });
    ans.push(low);
    ans
}

pub fn reorder_log_files(logs: Vec<String>) -> Vec<String> {
    fn is_digit(x: &String) -> bool {
        x.chars().rev().next().unwrap().is_digit(10)
    }

    fn log_cmp(x: &String, y: &String) -> Ordering {
        match (is_digit(x), is_digit(y)) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => {
                let x_split = x.split_ascii_whitespace().collect::<Vec<&str>>();
                let y_split = y.split_ascii_whitespace().collect::<Vec<&str>>();
                for i in 1..x_split.len().min(y_split.len()) {
                    match x_split[i].cmp(y_split[i]) {
                        Ordering::Greater => return Ordering::Greater,
                        Ordering::Less => return Ordering::Less,
                        Ordering::Equal => continue,
                    }
                }
                if x_split.len() < y_split.len() {
                    return Ordering::Less;
                } else if x_split.len() > y_split.len() {
                    return Ordering::Greater;
                }
                match x_split[0].cmp(y_split[0]) {
                    Ordering::Greater => return Ordering::Greater,
                    Ordering::Less => return Ordering::Less,
                    Ordering::Equal => Ordering::Equal,
                }
            }
        }
    }
    let mut logs = logs;
    logs.sort_by(|x, y| log_cmp(x, y));
    logs
}

pub fn min_distance_72(word1: String, word2: String) -> i32 {
    let word1 = word1.chars().collect::<Vec<char>>();
    let word2 = word2.chars().collect::<Vec<char>>();
    let (len1, len2) = (word1.len(), word2.len());
    if len1 * len2 == 0 {
        return len1.max(len2) as i32;
    }
    let mut dp = vec![vec![0; len1 + 1]; len2 + 1];
    for i in 0..=len1 {
        dp[0][i] = i as i32;
    }
    for j in 0..=len2 {
        dp[j][0] = j as i32;
    }
    for j in 1..=len2 {
        for i in 1..=len1 {
            if word1[i - 1] == word2[j - 1] {
                dp[j][i] = (dp[j - 1][i] + 1)
                    .min(dp[j][i - 1] + 1)
                    .min(dp[j - 1][i - 1]);
            } else {
                dp[j][i] = 1 + dp[j - 1][i].min(dp[j][i - 1]).min(dp[j - 1][i - 1]);
            }
        }
    }
    return dp[len2][len1];
}

pub fn one_edit_away(first: String, second: String) -> bool {
    let (len1, len2) = (first.len(), second.len());
    let mut first = first;
    let mut second = second;
    if len1 < len2 {
        let t = first.clone();
        first = second;
        second = t;
    }
    let (len1, len2) = (first.len(), second.len());
    let first = first.chars().collect::<Vec<char>>();
    let second = second.chars().collect::<Vec<char>>();
    if len1 - len2 >= 2 {
        return false;
    } else if len1 == len2 {
        let mut cnt = 0;
        for i in 0..len1 {
            if first[i] != second[i] {
                cnt += 1;
                if cnt >= 2 {
                    return false;
                }
            }
        }
        return true;
    } else {
        let (mut i, mut j) = (0, 0);
        while i < len1 && j < len2 {
            if first[i] != second[j] {
                i += 1;
            } else {
                i += 1;
                j += 1;
            }
            if i - j >= 2 {
                return false;
            }
        }
        return true;
    }
}

pub fn largest_triangle_area(points: Vec<Vec<i32>>) -> f64 {
    fn cal_area(a: Vec<i32>, b: Vec<i32>, c: Vec<i32>) -> f64 {
        return (a[0] * b[1] + b[0] * c[1] + c[0] * a[1] - a[0] * c[1] - b[0] * a[1] - c[0] * b[1])
            .abs() as f64
            / 2.0;
    }
    let mut ans: f64 = 0.0;
    for i in 0..points.len() {
        for j in i + 1..points.len() {
            for k in j + 1..points.len() {
                ans = ans.max(cal_area(
                    points[i].clone(),
                    points[j].clone(),
                    points[k].clone(),
                ));
            }
        }
    }
    ans
}

pub fn remove_anagrams(words: Vec<String>) -> Vec<String> {
    let mut words = words;
    fn is_allotopia(s1: &String, s2: &String) -> bool {
        if s1.len() != s2.len() {
            return false;
        }
        let mut s1 = s1.chars().collect::<Vec<char>>();
        let mut s2 = s2.chars().collect::<Vec<char>>();
        s1.sort();
        s2.sort();
        for i in 0..s1.len() {
            if s1[i] != s2[i] {
                return false;
            }
        }
        true
    }
    let mut del = vec![];
    for i in 1..words.len() {
        if is_allotopia(&words[i], &words[i - 1]) {
            del.push(i);
        }
    }
    del.iter().rev().for_each(|x| {
        words.remove(*x);
    });
    words
}

pub fn max_consecutive(bottom: i32, top: i32, special: Vec<i32>) -> i32 {
    let mut special = special;
    special.sort();
    let mut ans = special[0] - bottom;
    for i in 1..special.len() {
        ans = ans.max(special[i] - special[i - 1] - 1);
    }
    ans = ans.max(top - special[special.len() - 1]);
    ans
}

pub fn largest_combination(candidates: Vec<i32>) -> i32 {
    let mut cnt = vec![0; 24];
    for i in 0..candidates.len() {
        let mut x = candidates[i];
        let mut idx = 0;
        while x != 0 {
            cnt[idx] += x % 2;
            x /= 2;
            idx += 1;
        }
    }
    cnt.sort();
    cnt[cnt.len() - 1]
}

pub fn find_kth_number(m: i32, n: i32, k: i32) -> i32 {
    let (mut left, mut right) = (1, m * n);
    while left < right {
        let x = left + (right - left) / 2;
        let mut cnt = x / n * n;
        for i in x / n + 1..=m {
            cnt += x / i;
        }
        if cnt > k {
            right = x;
        } else {
            left = x + 1;
        }
    }
    return left;
}

pub fn longest_palindrome_subseq(s: String) -> i32 {
    let mut dp = vec![vec![0; s.len()]; s.len()];
    let n = s.len();
    let s = s.chars().collect::<Vec<char>>();
    for i in 0..=n - 1 {
        dp[i][i] = 1;
    }
    for i in (0..=n - 1).rev() {
        for j in i + 1..=n - 1 {
            if s[i] == s[j] {
                if j == i + 1 {
                    dp[i][j] = 2;
                    continue;
                }
                dp[i][j] = dp[i + 1][j - 1] + 2;
            } else {
                dp[i][j] = dp[i + 1][j].max(dp[i][j - 1]);
            }
        }
    }
    dp[0][n - 1]
}

pub fn min_moves2(nums: Vec<i32>) -> i32 {
    let n = nums.len();
    let mut nums = nums;
    nums.sort();
    let mid = {
        if n % 2 == 0 {
            (nums[n / 2] + nums[n / 2 - 1]) / 2
        } else {
            nums[n / 2]
        }
    };
    let mut ans = 0;
    nums.iter().for_each(|&x| {
        ans += (x - mid).abs();
    });
    ans
}

pub fn find_right_interval(intervals: Vec<Vec<i32>>) -> Vec<i32> {
    struct Node {
        index: usize,
        start: i32,
    }
    let n = intervals.len();
    let mut nodes = vec![];
    for i in 0..n {
        nodes.push(Node {
            index: i,
            start: intervals[i][0],
        })
    }
    nodes.sort_by_key(|x| x.start);
    fn bin_search(left: usize, right: usize, target: i32, nodes: &Vec<Node>) -> usize {
        if left < right {
            let mid = left + (right - left) / 2;
            if nodes[mid].start == target {
                return nodes[mid].index;
            } else if nodes[mid].start > target {
                return bin_search(left, mid, target, nodes);
            } else if nodes[mid].start < target {
                return bin_search(mid + 1, right, target, nodes);
            }
        }
        return nodes[left].index;
    }
    let mut ans: Vec<i32> = vec![];
    for i in 0..n {
        if intervals[i][1] > nodes[nodes.len() - 1].start {
            ans.push(-1);
            continue;
        }
        let idx = bin_search(0, n - 1, intervals[i][1], &nodes);
        ans.push(idx as i32);
    }
    ans
}

pub fn repeated_n_times(nums: Vec<i32>) -> i32 {
    let n = nums.len();
    loop {
        let x = random::<usize>() % n;
        let y = random::<usize>() % n;
        if x != y && nums[x] == nums[y] {
            return nums[x];
        }
    }
}

pub fn str_str(haystack: String, needle: String) -> i32 {
    if needle.len() == 0 {
        return 0;
    }
    match haystack.find(&needle) {
        None => -1,
        Some(x) => x as i32,
    }
}

pub fn search_range(nums: Vec<i32>, target: i32) -> Vec<i32> {
    if nums.len() == 0 {
        return vec![-1, -1];
    }
    let (mut left, mut right) = (0i32, nums.len() as i32 - 1);
    let mut li = nums.len() as i32;
    let mut ri = nums.len() as i32;
    while left <= right {
        let mid = left + (right - left) / 2;
        let mid = mid as usize;
        if nums[mid] >= target {
            li = mid as i32;
            right = mid as i32 - 1;
        } else {
            left = mid as i32 + 1;
        }
    }
    let (mut left, mut right) = (0i32, nums.len() as i32 - 1);
    while left <= right {
        let mid = left + (right - left) / 2;
        let mid = mid as usize;
        if nums[mid] > target {
            ri = mid as i32;
            right = mid as i32 - 1;
        } else {
            left = mid as i32 + 1;
        }
    }
    ri -= 1;
    if li <= ri
        && ri <= nums.len() as i32 - 1
        && nums[li as usize] == target
        && nums[ri as usize] == target
    {
        return vec![li, ri];
    }
    vec![-1, -1]
}

pub fn can_i_win(max_choosable_integer: i32, desired_total: i32) -> bool {
    if max_choosable_integer >= desired_total {
        return true;
    }
    if (max_choosable_integer + 1) * max_choosable_integer / 2 < desired_total {
        return false;
    }
    let mut dp = vec![None; 1 << max_choosable_integer];
    return dfs(0, max_choosable_integer, desired_total, &mut dp);
    fn dfs(
        index: usize,
        max_choosable_integer: i32,
        desired_total: i32,
        dp: &mut Vec<Option<bool>>,
    ) -> bool {
        if let Some(b) = dp[index] {
            return b;
        }
        for i in 1..=max_choosable_integer {
            let cur = 1 << (i - 1); // use bit position to represent number
            if cur & index != 0 {
                //cur has been chosen
                continue;
            }
            if i >= desired_total || !dfs(cur | index, max_choosable_integer, desired_total - i, dp)
            {
                dp[index] = Some(true);
                return true;
            }
        }
        dp[index] = Some(false);
        false
    }
}

pub fn percentage_letter(s: String, letter: char) -> i32 {
    let s = s.chars().collect::<Vec<char>>();
    let mut cnt = 0;
    let n = s.len();
    for i in 0..n {
        if s[i] == letter {
            cnt += 1;
        }
    }
    cnt * 100 / (n as i32)
}

pub fn maximum_bags(capacity: Vec<i32>, rocks: Vec<i32>, additional_rocks: i32) -> i32 {
    let mut remain: Vec<(i32, usize)> = vec![];
    for i in 0..capacity.len() {
        remain.push((capacity[i] - rocks[i], i));
    }
    remain.sort_by_key(|x| x.0);
    let mut additional_rocks = additional_rocks;
    let mut cnt = 0;
    for i in 0..remain.len() {
        if additional_rocks <= 0 {
            break;
        }
        if remain[i].0 == 0 {
            cnt += 1;
            continue;
        }
        if additional_rocks >= remain[i].0 {
            additional_rocks -= remain[i].0;
            remain[i].0 = 0;
            cnt += 1;
        }
    }
    cnt
}

pub fn minimum_lines(stock_prices: Vec<Vec<i32>>) -> i32 {
    if stock_prices.len() == 1 {
        return 1;
    } else if stock_prices.len() == 0 {
        return 0;
    }
    let mut cnt = stock_prices.len() - 1;
    let mut stock_prices = stock_prices.into_iter().collect::<Vec<Vec<i32>>>();
    stock_prices.sort_by_key(|x| x[0]);
    let mut lx = stock_prices[1][0] - stock_prices[0][0];
    let mut ly = stock_prices[1][1] - stock_prices[0][1];
    for i in 1..stock_prices.len() - 1 {
        let nx = stock_prices[i + 1][0] - stock_prices[i][0];
        let ny = stock_prices[i + 1][1] - stock_prices[i][1];
        if lx * ny == ly * nx {
            cnt -= 1;
        }
        lx = nx;
        ly = ny;
    }
    cnt as i32
}

pub fn cut_off_tree(forest: Vec<Vec<i32>>) -> i32 {
    struct Node {
        x: usize,
        y: usize,
        height: i32,
    }
    fn bfs(
        forest: &Vec<Vec<i32>>,
        sx: usize,
        sy: usize,
        tx: usize,
        ty: usize,
        row: usize,
        col: usize,
    ) -> i32 {
        if sx == tx && sy == ty {
            return 0;
        }
        let dir: Vec<Vec<i32>> = vec![vec![-1, 0], vec![0, -1], vec![0, 1], vec![1, 0]];
        let mut step = 0;
        let mut visited: Vec<Vec<bool>> = vec![vec![false; col]; row];
        let mut queue: Vec<usize> = vec![0; row * col];
        let mut head: usize = 0;
        let mut tail: usize = 0;
        queue[tail] = sx * col + sy;
        visited[sx][sy] = true;
        tail += 1;
        while head != tail {
            step += 1;
            let sz = tail - head;
            for _ in 0..sz {
                let cx = queue[head] / col;
                let cy = queue[head] % col;
                head += 1;
                for j in 0..4 {
                    let nx = cx as i32 + dir[j][0];
                    let ny = cy as i32 + dir[j][1];
                    if nx >= 0 && nx < row as i32 && ny >= 0 && ny < col as i32 {
                        if !visited[nx as usize][ny as usize]
                            && forest[nx as usize][ny as usize] > 0
                        {
                            if nx == tx as i32 && ny == ty as i32 {
                                return step;
                            }
                            visited[nx as usize][ny as usize] = true;
                            queue[tail] = nx as usize * col + ny as usize;
                            tail += 1;
                        }
                    }
                }
            }
        }
        return -1;
    }
    let mut nodes = vec![];
    let row = forest.len();
    let col = forest[0].len();
    for i in 0..row {
        for j in 0..col {
            if forest[i][j] > 1 {
                nodes.push(Node {
                    x: i,
                    y: j,
                    height: forest[i][j],
                });
            }
        }
    }
    let mut step = 0;
    nodes.sort_by_key(|x| x.height);
    let mut cx = 0;
    let mut cy = 0;
    for i in 0..nodes.len() {
        let s = bfs(&forest, cx, cy, nodes[i].x, nodes[i].y, row, col);
        if s == -1 {
            return -1;
        }
        step += s;
        cx = nodes[i].x;
        cy = nodes[i].y;
    }
    step
}

pub fn multiply(num1: String, num2: String) -> String {
    if (num1.len() == 1 && num1.parse::<i32>().unwrap() == 0)
        || (num2.len() == 1 && num2.parse::<i32>().unwrap() == 0)
    {
        return '0'.to_string();
    }
    let n1 = num1.len();
    let n2 = num2.len();
    let nums1;
    let nums2;
    nums1 = num1.chars().rev().collect::<Vec<char>>();
    nums2 = num2.chars().rev().collect::<Vec<char>>();
    let mut ans = vec![0; n1 + n2];
    for j in 0..n2 {
        for i in 0..n1 {
            let x = nums1[i].to_digit(10).unwrap();
            let y = nums2[j].to_digit(10).unwrap();
            let low = x * y % 10;
            let high = x * y / 10;
            ans[i + j] += low;
            ans[i + j + 1] += high;
        }
    }
    let mut c = 0;
    for i in 0..ans.len() {
        ans[i] += c;
        if ans[i] >= 10 {
            c = ans[i] / 10;
            ans[i] = ans[i] % 10;
        } else {
            c = 0;
        }
    }
    if ans[n1 + n2 - 1] == 0 {
        ans.drain(n1 + n2 - 1..);
    }
    ans.iter()
        .rev()
        .map(|x| x.to_string().chars().next().unwrap())
        .collect()
}

pub fn is_valid_sudoku(board: Vec<Vec<char>>) -> bool {
    let mut rows = vec![vec![0; 9]; 9];
    let mut cols = vec![vec![0; 9]; 9];
    let mut grid = vec![vec![vec![0; 9]; 3]; 3];
    for i in 0..board.len() {
        for j in 0..board[0].len() {
            if board[i][j] != '.' {
                let x = board[i][j].to_digit(10).unwrap();
                rows[i][x as usize - 1] += 1;
                cols[j][x as usize - 1] += 1;
                grid[i / 3][j / 3][x as usize - 1] += 1;
                if rows[i][x as usize - 1] > 1
                    || cols[j][x as usize - 1] > 1
                    || grid[i / 3][j / 3][x as usize - 1] > 1
                {
                    return false;
                }
            }
        }
    }
    true
}

pub fn set_zeroes(matrix: &mut Vec<Vec<i32>>) {
    let mut nodes = vec![];
    for i in 0..matrix.len() {
        for j in 0..matrix[0].len() {
            if matrix[i][j] == 0 {
                nodes.push((i, j));
            }
        }
    }
    let m = matrix.len();
    let n = matrix[0].len();
    for i in 0..nodes.len() {
        let x = nodes[i].0;
        let y = nodes[i].1;
        for j in 0..m {
            matrix[j][y] = 0;
        }
        for k in 0..n {
            matrix[x][k] = 0;
        }
    }
}

pub fn find_substring_in_wrapround_string(p: String) -> i32 {
    let p = p
        .chars()
        .map(|c| c as usize - 'a' as usize)
        .collect::<Vec<usize>>();
    let mut k: i32 = 1;
    let mut dp: Vec<i32> = vec![0; 26];
    dp[p[0]] = 1;
    for i in 1..p.len() {
        if p[i] == p[i - 1] + 1 {
            k += 1;
        } else {
            k = 1;
        }
        dp[p[i]] = dp[p[i]].max(k);
    }
    dp.iter().sum()
}

pub fn maximal_rectangle(matrix: Vec<Vec<char>>) -> i32 {
    let mut left = vec![vec![0; matrix[0].len()]; matrix.len()];
    for i in 0..matrix.len() {
        for j in 0..matrix[0].len() {
            if j == 0 {
                left[i][j] = matrix[i][j].to_digit(10).unwrap();
            } else if matrix[i][j] == '1' {
                left[i][j] = left[i][j - 1] + 1;
            } else {
                left[i][j] = 0;
            }
        }
    }
    let mut ans = 0;
    for i in 0..matrix.len() {
        for j in 0..matrix[0].len() {
            let mut max = left[i][j];
            let mut min = left[i][j];
            if i == 0 {
                ans = ans.max(max);
                continue;
            }
            for k in (0..=i - 1).rev() {
                min = min.min(left[k][j]);
                max = max.max(min * (i as u32 - k as u32 + 1));
            }
            ans = ans.max(max);
        }
    }
    ans as i32
}

pub fn generate(num_rows: i32) -> Vec<Vec<i32>> {
    let mut ans = vec![vec![]; num_rows as usize];
    for i in 0..num_rows as usize {
        for j in 0..=i {
            if i == 0 || j == 0 || j == i {
                ans[i].push(1);
            } else {
                let x = ans[i - 1][j - 1];
                let y = ans[i - 1][j];
                ans[i].push(x + y);
            }
        }
    }
    ans
}

pub fn get_row(row_index: i32) -> Vec<i32> {
    let mut ans: Vec<i32> = vec![];
    ans.push(1);
    for i in 1..=row_index as usize {
        let prev = ans[i - 1];
        ans.push((prev as i64 * (row_index as i64 - i as i64 + 1) / i as i64) as i32);
    }
    ans
}

pub fn falling_squares(positions: Vec<Vec<i32>>) -> Vec<i32> {
    let mut ans = vec![];
    for i in 0..positions.len() {
        let l1 = positions[i][0];
        let r1 = positions[i][0] + positions[i][1] - 1;
        ans.push(positions[i][1]);
        for j in 0..i {
            let l2 = positions[j][0];
            let r2 = positions[j][1] + positions[j][0] - 1;
            if l2 <= r1 && r2 >= l1 {
                ans[i] = ans[i].max(ans[j] + positions[i][1]);
            }
        }
    }
    for i in 1..ans.len() {
        ans[i] = ans[i].max(ans[i - 1]);
    }
    ans
}

pub fn is_scramble(s1: String, s2: String) -> bool {
    let s1 = s1.chars().collect::<Vec<char>>();
    let s2 = s2.chars().collect::<Vec<char>>();
    fn check_equal(s1: &Vec<char>, s2: &Vec<char>, i1: usize, i2: usize, len: usize) -> bool {
        for i in 0..len {
            if s1[i1 + i] != s2[i2 + i] {
                return false;
            }
        }
        return true;
    }
    fn check_letter(s1: &Vec<char>, s2: &Vec<char>, i1: usize, i2: usize, len: usize) -> bool {
        let mut h1 = HashMap::new();
        let mut h2 = HashMap::new();
        for i in 0..len {
            let e1 = h1.entry(s1[i1 + i]).or_insert(0);
            *e1 += 1;
            let e2 = h2.entry(s2[i2 + i]).or_insert(0);
            *e2 += 1;
        }
        for (k, v) in h1.iter() {
            let v2 = h2.get(k);
            if v2.is_none() || *v != *v2.unwrap() {
                return false;
            }
        }
        true
    }
    let mut memo: Vec<Vec<Vec<i32>>> = vec![vec![vec![0; 31]; 30]; 30];
    fn dfs(
        s1: &Vec<char>,
        s2: &Vec<char>,
        i1: usize,
        i2: usize,
        len: usize,
        memo: &mut Vec<Vec<Vec<i32>>>,
    ) -> bool {
        if memo[i1][i2][len] == 1 {
            return true;
        } else if memo[i1][i2][len] == -1 {
            return false;
        }
        if check_equal(s1, s2, i1, i2, len) {
            memo[i1][i2][len] = 1;
            return true;
        }
        if !check_letter(s1, s2, i1, i2, len) {
            memo[i1][i2][len] = -1;
            return false;
        }
        for i in 1..len {
            if dfs(s1, s2, i1, i2, i, memo) && dfs(s1, s2, i1 + i, i2 + i, len - i, memo) {
                memo[i1][i2][len] = 1;
                return true;
            }
            if dfs(s1, s2, i1, i2 + len - i, i, memo) && dfs(s1, s2, i1 + i, i2, len - i, memo) {
                memo[i1][i2][len] = 1;
                return true;
            }
        }
        memo[i1][i2][len] = -1;
        false
    }
    dfs(&s1, &s2, 0, 0, s1.len(), &mut memo)
}

pub fn is_interleave(s1: String, s2: String, s3: String) -> bool {
    let n1 = s1.len();
    let n2 = s2.len();
    let n3 = s3.len();
    if n3 != n1 + n2 {
        return false;
    }
    let mut dp = vec![vec![false; n2 + 1]; n1 + 1];
    let s1 = s1.chars().collect::<Vec<char>>();
    let s2 = s2.chars().collect::<Vec<char>>();
    let s3 = s3.chars().collect::<Vec<char>>();
    dp[0][0] = true;
    for i in 0..=n1 {
        for j in 0..=n2 {
            let p = i + j - 1;
            if i > 0 {
                dp[i][j] |= dp[i - 1][j] && s1[i - 1] == s3[p];
            }
            if j > 0 {
                dp[i][j] |= dp[i][j - 1] && s2[j - 1] == s3[p]
            }
        }
    }
    dp[n1][n2]
}

pub fn find_closest(words: Vec<String>, word1: String, word2: String) -> i32 {
    // use std::collections::HashMap;
    // let mut hp : HashMap<String, Vec<usize>> = HashMap::new();
    // for i in 0..words.len(){
    //     let e = hp.entry(words[i].clone()).or_insert(vec![]);
    //     (*e).push(i);
    // }
    // let w1 = hp.get(&word1).unwrap();
    // let w2 = hp.get(&word2).unwrap();
    // let mut ans = i32::MAX;
    // for i in 0..w1.len(){
    //     let mut temp = i32::MAX;
    //     for j in 0..w2.len(){
    //         let t = (w1[i] as i32 - w2[j] as i32).abs();
    //         if t <=temp {
    //             temp = t;
    //         }else{
    //             break;
    //         }
    //     }
    //     ans = ans.min(temp);
    // }
    // ans
    let mut ans = i32::MAX;
    let mut i1 = -1;
    let mut i2 = -1;
    for i in 0..words.len() {
        if words[i] == word1 {
            i1 = i as i32;
        } else if words[i] == word2 {
            i2 = i as i32;
        }
        if i1 >= 0 && i2 >= 0 {
            ans = ans.min((i1 - i2).abs());
        }
    }
    ans
}

pub fn num_distinct(s: String, t: String) -> i32 {
    let mut dp = vec![vec![0; t.len() + 1]; s.len() + 1];
    for i in 0..=s.len() {
        dp[i][t.len()] = 1;
    }
    let m = s.len();
    let n = t.len();
    if m < n {
        return 0;
    }
    let s = s.chars().collect::<Vec<char>>();
    let t = t.chars().collect::<Vec<char>>();
    for i in (0..s.len()).rev() {
        for j in (0..t.len()).rev() {
            if s[i] == t[j] {
                dp[i][j] = dp[i + 1][j] + dp[i + 1][j + 1];
            } else {
                dp[i][j] = dp[i + 1][j];
            }
        }
    }
    dp[0][0]
}

pub fn remove_outer_parentheses(s: String) -> String {
    let s = s.chars().collect::<Vec<char>>();
    let mut ans: Vec<char> = vec![];
    let mut cnt = 0;
    let mut idx = 0;
    for i in 0..s.len() {
        if s[i] == '(' {
            cnt += 1;
        } else if s[i] == ')' {
            cnt -= 1;
        }
        if cnt == 0 {
            for j in idx + 1..i {
                ans.push(s[j]);
            }
            idx = i + 1;
        }
    }
    ans.iter().collect()
}

pub fn valid_ip_address(query_ip: String) -> String {
    let ipv4: Vec<&str> = query_ip.split('.').collect();
    let ipv6: Vec<&str> = query_ip.split(":").collect();
    if ipv4.len() != 4 && ipv6.len() != 8 {
        return "Neither".to_string();
    }
    fn check_valid(s: &str, ipv4: bool) -> bool {
        let ss = s.to_string();
        if ipv4 {
            let n;
            match ss.parse::<i32>() {
                Err(_) => return false,
                Ok(x) => n = x,
            };
            if n != 0 {
                let sss = ss.chars().collect::<Vec<char>>();
                if sss[0] == '0' {
                    return false;
                }
            }
            if n == 0 && ss.len() != 1 {
                return false;
            } else if n >= 0 && n < 256 {
                return true;
            }
        } else {
            if ss.len() > 4 || ss.len() == 0 {
                return false;
            }
            let mut flag = true;
            ss.chars().for_each(|c| match c {
                '0'..='9' => {}
                'a'..='f' => {}
                'A'..='F' => {}
                _ => flag = false,
            });
            if flag {
                return true;
            }
        }
        false
    }
    if ipv4.len() == 4 {
        for s in ipv4 {
            if !check_valid(s, true) {
                return "Neither".to_string();
            }
        }
        return "IPv4".to_string();
    } else if ipv6.len() == 8 {
        for s in ipv6 {
            if !check_valid(s, false) {
                return "Neither".to_string();
            }
        }
        return "IPv6".to_string();
    }
    "Neither".to_string()
}

pub fn rearrange_characters(s: String, target: String) -> i32 {
    let mut ans = i32::MAX;
    let mut hp1 = HashMap::new();
    s.chars().for_each(|c| {
        let e = hp1.entry(c).or_insert(0);
        *e += 1;
    });
    let mut hp2 = HashMap::new();
    target.chars().for_each(|c| {
        let e = hp2.entry(c).or_insert(0);
        *e += 1;
    });
    for (k, &v) in hp2.iter() {
        let v1 = match hp1.get(k) {
            Some(x) => *x,
            None => 0,
        };
        if v1 == 0 {
            return 0;
        }
        let t = v1 / v;
        ans = ans.min(t);
    }
    ans
}

pub fn total_steps(nums: Vec<i32>) -> i32 {
    let mut step = 0;
    if nums.len() == 1 {
        return 0;
    }
    let mut nums = nums;
    let mut idx = vec![];
    for i in 1..nums.len() {
        if nums[i] < nums[i - 1] {
            idx.push(i);
        }
    }
    while idx.len() != 0 {
        step += 1;
        for j in (0..idx.len()).rev() {
            nums.remove(idx[j]);
        }
        idx.drain(..);
        for i in 1..nums.len() {
            if nums[i] < nums[i - 1] {
                idx.push(i);
            }
        }
    }
    step
}

pub fn partition_131(s: String) -> Vec<Vec<String>> {
    let mut ans = vec![];
    let s = s.chars().collect::<Vec<char>>();
    let mut dp = vec![vec![true; s.len()]; s.len()];
    for i in (0..s.len()).rev() {
        for j in i + 1..s.len() {
            dp[i][j] = s[i] == s[j] && dp[i + 1][j - 1];
        }
    }
    let n = s.len();
    let mut tmp = vec![];
    fn dfs(
        s: &Vec<char>,
        idx: usize,
        dp: &Vec<Vec<bool>>,
        n: usize,
        tmp: &mut Vec<String>,
        ans: &mut Vec<Vec<String>>,
    ) {
        if idx == n {
            ans.push(tmp.to_owned());
        }
        for j in idx..n {
            if dp[idx][j] {
                tmp.push(s[idx..=j].to_owned().iter().collect());
                dfs(s, j + 1, dp, n, tmp, ans);
                tmp.pop();
            }
        }
    }
    dfs(&s, 0, &dp, n, &mut tmp, &mut ans);
    ans
}

pub fn alien_order(words: Vec<String>) -> String {
    let mut indegree = vec![0; 26];
    let mut queue = vec![];
    let mut map = vec![vec![-1; 26]; 26];
    let mut valid = true;
    let mut vq = vec![false; 26];
    fn add_edge(
        w1: &Vec<char>,
        w2: &Vec<char>,
        map: &mut Vec<Vec<i32>>,
        indegree: &mut Vec<i32>,
    ) -> bool {
        let l1 = w1.len();
        let l2 = w2.len();
        let len = l1.min(l2);
        let mut idx = 0;
        for _ in 0..len {
            let c1 = w1[idx];
            let c2 = w2[idx];
            let x = (c1 as u8 - 'a' as u8) as usize;
            let y = (c2 as u8 - 'a' as u8) as usize;
            if c1 != c2 {
                if map[x][y] == -1 {
                    map[x][y] = 1;
                    indegree[y] += 1;
                }
                break;
            }
            idx += 1;
        }
        if idx == len && l1 > l2 {
            return false;
        }
        true
    }
    let alphabet = vec![
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
        's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    ];
    let words = words
        .iter()
        .map(|s| s.chars().collect::<Vec<char>>())
        .collect::<Vec<Vec<char>>>();
    for i in 1..words.len() {
        if !valid {
            break;
        }
        valid = add_edge(&words[i - 1], &words[i], &mut map, &mut indegree);
    }
    for i in 0..words.len() {
        for j in 0..words[i].len() {
            let x = (words[i][j] as u8 - 'a' as u8) as usize;
            vq[x] = true;
        }
    }
    let n = vq
        .iter()
        .fold(0, |acc, b| if *b == true { acc + 1 } else { acc });
    let mut head = 0;
    let mut tail = 0;
    let mut ans: Vec<char> = vec![];
    for i in 0..indegree.len() {
        if indegree[i] == 0 && vq[i] == true {
            queue.push(i);
            tail += 1;
        }
    }
    while head != tail {
        let x = queue[head];
        head += 1;
        ans.push(alphabet[x]);
        for i in 0..map[x].len() {
            if map[x][i] == 1 {
                indegree[i] -= 1;
                if indegree[i] == 0 {
                    queue.push(i);
                    tail += 1;
                }
            }
        }
    }
    if !valid || ans.len() != n {
        return "".to_owned();
    }
    ans.iter().collect()
}

pub fn makesquare(matchsticks: Vec<i32>) -> bool {
    let total = matchsticks.iter().sum::<i32>();
    if total % 4 != 0 {
        return false;
    }
    let ave = total / 4;
    let mut edges = vec![0; 4];
    fn dfs(matchsticks: &Vec<i32>, idx: usize, ave: i32, edges: &mut Vec<i32>) -> bool {
        if idx == matchsticks.len() {
            return true;
        }
        for i in 0..4 {
            edges[i] += matchsticks[idx];
            if edges[i] <= ave && dfs(matchsticks, idx + 1, ave, edges) {
                return true;
            }
            edges[i] -= matchsticks[idx];
        }
        return false;
    }
    let mut matchsticks = matchsticks;
    matchsticks.sort_by(|a, b| b.cmp(a));
    dfs(&matchsticks, 0, ave, &mut edges)
}

pub fn consecutive_numbers_sum(n: i32) -> i32 {
    fn is_consecutive(k: i32, n: i32) -> bool {
        if k % 2 == 1 && n % k == 0 {
            return true;
        } else if k % 2 == 0 && 2 * n % k == 0 && n % k != 0 {
            return true;
        }
        false
    }
    let mut ans = 0;
    let bound = n * 2;
    let mut k = 1;
    while k * (k + 1) <= bound {
        if is_consecutive(k, n) {
            ans += 1;
        }
        k += 1;
    }
    ans
}

pub fn num_unique_emails(emails: Vec<String>) -> i32 {
    let mut ans = 0;
    let mut hp = HashMap::new();
    for e in emails {
        let s = e.split('@').collect::<Vec<&str>>();
        let mut local = s[0]
            .to_string()
            .chars()
            .filter(|c| *c != '.')
            .collect::<String>();
        match local.find('+') {
            Some(i) => {
                let _ = local.drain(i..);
            }
            None => {}
        };
        let ns = local + "@" + s[1];
        let en = hp.entry(ns).or_insert(0);
        *en += 1;
    }
    for (_, _) in hp.iter() {
        ans += 1;
    }
    ans
}

pub fn max_product(nums: Vec<i32>) -> i32 {
    let mut fmax = nums.clone();
    let mut fmin = nums.clone();
    nums.iter()
        .enumerate()
        .skip(1)
        .fold(nums[0], |acc, (i, x)| {
            fmax[i] = (fmax[i - 1] * x).max((fmin[i - 1] * x).max(*x));
            fmin[i] = (fmax[i - 1] * x).min((fmin[i - 1] * x).min(*x));
            acc.max(fmax[i])
        })
}

pub fn remove_boxes(boxes: Vec<i32>) -> i32 {
    fn cal(boxes: &Vec<i32>, l: i32, r: i32, k: i32, dp: &mut Vec<Vec<Vec<i32>>>) -> i32 {
        if l > r {
            return 0;
        }
        if dp[l as usize][r as usize][k as usize] == 0 {
            dp[l as usize][r as usize][k as usize] =
                cal(boxes, l, r - 1, 0, dp) + (k + 1) * (k + 1);
            for i in l..r {
                if boxes[i as usize] == boxes[r as usize] {
                    dp[l as usize][r as usize][k as usize] = dp[l as usize][r as usize][k as usize]
                        .max(cal(boxes, l, i, k + 1, dp) + cal(boxes, i + 1, r - 1, 0, dp));
                }
            }
        }
        return dp[l as usize][r as usize][k as usize];
    }
    let mut dp = vec![vec![vec![0; boxes.len()]; boxes.len()]; boxes.len()];
    return cal(&boxes, 0, boxes.len() as i32 - 1, 0, &mut dp);
}

pub fn min_eating_speed(piles: Vec<i32>, h: i32) -> i32 {
    let max = piles.iter().max().unwrap().to_owned();
    let mut l = 1;
    let mut r = max;
    let mut ans = r;
    fn get_time(piles: &Vec<i32>, speed: i32) -> i64 {
        let mut t = 0;
        for pile in piles {
            t += ((pile + speed - 1) / speed) as i64;
        }
        t
    }
    while l < r {
        let speed = l + (r - l) / 2;
        let t = get_time(&piles, speed);
        if t <= h as i64 {
            ans = speed;
            r = speed;
        } else {
            l = speed + 1;
        }
    }
    ans
}

pub fn is_boomerang(points: Vec<Vec<i32>>) -> bool {
    let (x1, x2, x3) = (points[0][0], points[1][0], points[2][0]);
    let (y1, y2, y3) = (points[0][1], points[1][1], points[2][1]);
    if (x1 == x2 && y1 == y2) || (x1 == x3 && y1 == y3) || (x2 == x3 && y2 == y3) {
        return false;
    }
    if (y2 - y1) * (x3 - x2) == (y3 - y2) * (x2 - x1) {
        return false;
    }
    return true;
}

pub fn count_palindromic_subsequences(s: String) -> i32 {
    let s = s.chars().collect::<Vec<char>>();
    let n = s.len();
    let mut dp = vec![vec![vec![0; n]; n]; 4];
    for i in 0..n {
        let k = (s[i] as u8 - 'a' as u8) as usize;
        dp[k][i][i] = 1;
    }
    for len in 2..=n {
        for i in 0..=n - len {
            let j = i + len - 1;
            for k in 0..4 {
                if s[i] == s[j] && s[i] as u8 == 'a' as u8 + k {
                    let mut sum: i64 = 0;
                    for p in 0..4 {
                        sum = (sum + dp[p][i + 1][j - 1] % 1000_000_007) % 1000_000_007;
                    }
                    dp[k as usize][i][j] = (2 + sum) % 1000_000_007;
                } else if s[i] as u8 == 'a' as u8 + k && s[j] as u8 != 'a' as u8 + k {
                    dp[k as usize][i][j] = dp[k as usize][i][j - 1];
                } else if s[i] as u8 != 'a' as u8 + k && s[j] as u8 == 'a' as u8 + k {
                    dp[k as usize][i][j] = dp[k as usize][i + 1][j];
                } else {
                    dp[k as usize][i][j] = dp[k as usize][i + 1][j - 1];
                }
            }
        }
    }
    let mut ans = 0;
    for i in 0..4 {
        ans = (ans + dp[i][0][n - 1]) % 1000_000_007;
    }
    ans as i32
}

pub fn min_flips_mono_incr(s: String) -> i32 {
    let s = s.chars().collect::<Vec<char>>();
    let mut dp = vec![vec![0; 2]; s.len()];
    match s[0] {
        '0' => dp[0][1] = 1,
        '1' => dp[0][0] = 1,
        _ => {}
    };
    for i in 1..s.len() {
        dp[i][0] = dp[i - 1][0] + if s[i] == '1' { 1 } else { 0 };
        dp[i][1] = dp[i - 1][0].min(dp[i - 1][1]) + if s[i] == '0' { 1 } else { 0 };
    }
    dp[s.len() - 1][0].min(dp[s.len() - 1][1])
}

pub fn find_and_replace_pattern(words: Vec<String>, pattern: String) -> Vec<String> {
    let pattern = pattern.chars().collect::<Vec<char>>();
    let mut ans = vec![];
    for word in words {
        let word = word.chars().collect::<Vec<char>>();
        let mut hp = HashMap::new();
        let mut hp_rev = HashMap::new();
        let mut flag = true;
        for i in 0..word.len() {
            match hp.get(&word[i]) {
                None => {
                    hp.insert(word[i], pattern[i]);
                    if hp_rev.get(&pattern[i]).is_none() {
                        hp_rev.insert(pattern[i], word[i]);
                    } else {
                        let c = hp_rev.get(&pattern[i]).unwrap();
                        if *c != word[i] {
                            flag = false;
                            break;
                        }
                    }
                }
                Some(c) => {
                    if *c != pattern[i] {
                        flag = false;
                        break;
                    }
                }
            }
        }
        if flag {
            ans.push(word.iter().collect::<String>());
        }
    }
    ans
}

pub fn calculate_tax(brackets: Vec<Vec<i32>>, income: i32) -> f64 {
    let mut ans: f64 = 0.0;
    for i in 0..brackets.len() {
        if income >= brackets[i][0] {
            if i == 0 {
                ans += brackets[i][0] as f64 * brackets[i][1] as f64 / 100.0;
            } else {
                ans += (brackets[i][0] - brackets[i - 1][0]) as f64 * brackets[i][1] as f64 / 100.0;
            }
        } else {
            if i == 0 {
                ans += income as f64 * brackets[i][1] as f64 / 100.0;
            } else {
                ans += (income - brackets[i - 1][0]) as f64 * brackets[i][1] as f64 / 100.0;
            }
            break;
        }
    }
    ans
}

pub fn min_path_cost(grid: Vec<Vec<i32>>, move_cost: Vec<Vec<i32>>) -> i32 {
    let m = grid.len();
    let n = grid[0].len();
    let mut dp = vec![vec![0; n]; m];
    for i in 0..n {
        dp[0][i] = grid[0][i];
    }
    for i in 1..m {
        for j in 0..n {
            let mut min = i32::MAX;
            for k in 0..n {
                min = min.min(dp[i - 1][k] + move_cost[grid[i - 1][k] as usize][j]);
            }
            dp[i][j] = min + grid[i][j];
        }
    }
    let mut ans = i32::MAX;
    for i in 0..n {
        ans = ans.min(dp[m - 1][i]);
    }
    ans
}

pub fn distribute_cookies(cookies: Vec<i32>, k: i32) -> i32 {
    fn dfs(bk: &mut Vec<i32>, cookies: &Vec<i32>, idx: usize, k: i32) -> i32 {
        if idx == cookies.len() {
            return bk.iter().max_by_key(|x| **x).unwrap().to_owned();
        } else {
            let mut temp = i32::MAX;
            for i in 0..k {
                bk[i as usize] += cookies[idx];
                temp = temp.min(dfs(bk, cookies, idx + 1, k));
                bk[i as usize] -= cookies[idx];
            }
            return temp;
        }
    }
    let mut bk = vec![0; k as usize];
    return dfs(&mut bk, &cookies, 0, k);
}

pub fn distinct_names(ideas: Vec<String>) -> i64 {
    let mut hp = HashMap::new();
    for s in &ideas {
        hp.insert(s.clone(), 1);
    }
    let mut ans: i64 = 0;
    for i in 0..ideas.len() {
        for j in i + 1..ideas.len() {
            let mut ia = ideas[i].clone().chars().collect::<Vec<char>>();
            let mut ib = ideas[j].clone().chars().collect::<Vec<char>>();
            let c = ia[0];
            ia[0] = ib[0];
            ib[0] = c;
            let ia = ia.iter().collect::<String>();
            let ib = ib.iter().collect::<String>();
            if let Some(_) = hp.get(&ia) {
                continue;
            }
            if let Some(_) = hp.get(&ib) {
                continue;
            }
            ans += 1;
        }
    }
    ans
}

pub fn height_checker(heights: Vec<i32>) -> i32 {
    let mut new_heights = heights.clone();
    new_heights.sort();
    new_heights
        .iter()
        .zip(heights.iter())
        .fold(0, |acc, (a, b)| acc + if a != b { 1 } else { 0 })
}

pub fn find_diagonal_order(mat: Vec<Vec<i32>>) -> Vec<i32> {
    let mut mid: Vec<Vec<i32>> = vec![];
    let m = mat.len();
    let n = mat[0].len();
    if m == 1 || n == 1 {
        let mut ans = vec![];
        for i in 0..m {
            for j in 0..n {
                ans.push(mat[i][j]);
            }
        }
        return ans;
    }
    // scan by first row
    for i in 0..n {
        let mut tmp = vec![];
        for j in 0..=i.min(m - 1) {
            tmp.push(mat[j][i - j]);
        }
        mid.push(tmp);
    }
    // scan by last column
    for j in 1..m {
        let mut tmp = vec![];
        let mut t = j;
        let mut i = n as i32 - 1;
        while i >= 0 && t < m {
            tmp.push(mat[t][i as usize]);
            t += 1;
            i -= 1;
        }
        mid.push(tmp);
    }
    let mut ans = vec![];
    for i in 0..mid.len() {
        if i % 2 == 1 {
            ans.append(&mut mid[i]);
        } else {
            for j in mid[i].iter().rev() {
                ans.push(*j);
            }
        }
    }
    ans
}

pub fn smallest_distance_pair(nums: Vec<i32>, k: i32) -> i32 {
    let mut nums = nums;
    nums.sort();
    fn bin_search(nums: &Vec<i32>, end: i32, target: i32) -> i32 {
        let (mut l, mut r) = (0, end - 1);
        while l <= r {
            let mid = l + (r - l) / 2;
            if nums[mid as usize] < target {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        l
    }
    let range = nums[nums.len() - 1] - nums[0];
    let (mut l, mut r) = (0i32, range);
    while l <= r {
        let mut cnt = 0;
        let mid = l + (r - l) / 2;
        for j in 0..nums.len() {
            let idx = bin_search(&nums, j as i32, nums[j] - mid as i32);
            cnt += j as i32 - idx;
        }
        if cnt < k {
            l = mid + 1;
        } else {
            r = mid - 1;
        }
    }
    l as i32
}

pub fn solve_sudoku(board: &mut Vec<Vec<char>>) {
    let mut lines = vec![vec![false; 9]; 9];
    let mut column = vec![vec![false; 9]; 9];
    let mut space = vec![vec![vec![false; 9]; 3]; 3];
    let mut omit = vec![];
    for i in 0..board.len() {
        for j in 0..board[0].len() {
            if board[i][j] == '.' {
                omit.push(vec![i, j]);
            } else {
                let n = board[i][j] as u8 - '1' as u8;
                lines[i][n as usize] = true;
                column[j][n as usize] = true;
                space[i / 3][j / 3][n as usize] = true;
            }
        }
    }
    fn dfs(
        board: &mut Vec<Vec<char>>,
        lines: &mut Vec<Vec<bool>>,
        column: &mut Vec<Vec<bool>>,
        space: &mut Vec<Vec<Vec<bool>>>,
        omit: &mut Vec<Vec<usize>>,
        pos: usize,
        valid: &mut bool,
    ) {
        if pos == omit.len() {
            *valid = true;
            return;
        }
        let (i, j) = (omit[pos][0], omit[pos][1]);
        for num in 0..9 {
            if *valid == true {
                break;
            }
            if !lines[i][num] && !column[j][num] && !space[i / 3][j / 3][num] {
                lines[i][num] = true;
                column[j][num] = true;
                space[i / 3][j / 3][num] = true;
                board[i][j] = ('1' as u8 + num as u8) as char;
                dfs(board, lines, column, space, omit, pos + 1, valid);
                lines[i][num] = false;
                column[j][num] = false;
                space[i / 3][j / 3][num] = false;
            }
        }
    }
    let mut valid = false;
    dfs(
        board,
        &mut lines,
        &mut column,
        &mut space,
        &mut omit,
        0,
        &mut valid,
    );
}

pub fn find_pairs(nums: Vec<i32>, k: i32) -> i32 {
    let mut nums = nums;
    nums.sort();
    let mut j = 0;
    let mut res = 0;
    for i in 0..nums.len() {
        if i != 0 && nums[i] == nums[i - 1] {
            continue;
        }
        while j < nums.len() && (nums[j] - nums[i] < k || j <= i) {
            j += 1;
        }
        if j < nums.len() && nums[j] - nums[i] == k {
            res += 1;
        }
    }
    res
}

pub fn duplicate_zeros(arr: &mut Vec<i32>) {
    let mut i: i32 = -1;
    let mut top = 0;
    while top < arr.len() {
        i += 1;
        if arr[i as usize] != 0 {
            top += 1;
        } else {
            top += 2;
        }
    }
    let mut j: i32 = arr.len() as i32 - 1;
    if top == arr.len() + 1 {
        arr[j as usize] = 0;
        j -= 1;
        i -= 1;
    }
    while j >= 0 {
        arr[j as usize] = arr[i as usize];
        j -= 1;
        if arr[i as usize] == 0 {
            arr[j as usize] = 0;
            j -= 1;
        }
        i -= 1;
    }
}

pub fn greatest_letter(s: String) -> String {
    let s = s.chars().collect::<Vec<char>>();
    let mut hp = HashMap::new();
    let mut ans = HashMap::new();
    for c in s.iter() {
        if c.is_uppercase() {
            let cl = c.to_lowercase().to_string();
            if hp.get(&cl).is_some() {
                ans.insert(c.to_string(), 1);
            } else {
                hp.insert(c.to_string(), 1);
            }
        } else {
            let cu = c.to_uppercase().to_string();
            if hp.get(&cu).is_some() {
                ans.insert(cu, 1);
            } else {
                hp.insert(c.to_string(), 1);
            }
        }
    }
    if ans.is_empty() {
        return "".to_string();
    }
    let mut ret = String::from("");
    for (k, _) in ans.iter() {
        if ret.is_empty() {
            ret = k.clone();
        } else {
            if k.cmp(&ret).is_gt() {
                ret = k.clone();
            }
        }
    }
    ret
}

pub fn minimum_numbers(num: i32, k: i32) -> i32 {
    if num == 0 {
        return 0;
    } else if num % 2 == 1 && k % 2 == 0 {
        return -1;
    } else if num % 10 != 0 && k == 0 {
        return -1;
    } else if num % 10 == k {
        return 1;
    } else if num < k {
        return -1;
    }
    for i in 1..=10 {
        let t = k * i;
        if t % 10 == num % 10 && t <= num {
            return t;
        }
    }
    -1
}

pub fn defang_i_paddr(address: String) -> String {
    address.replace(".", "[.]")
}

pub fn number_of_lines(widths: Vec<i32>, s: String) -> Vec<i32> {
    s.chars()
        .map(|c| c as usize - 'a' as usize)
        .collect::<Vec<_>>()
        .iter()
        .fold(vec![1, 0], |acc, &x| {
            if acc[1] + widths[x] > 100 {
                vec![acc[0] + 1, widths[x]]
            } else {
                vec![acc[0], acc[1] + widths[x]]
            }
        })
}

pub fn min_cost(costs: Vec<Vec<i32>>) -> i32 {
    let n = costs.len();
    let mut dp = vec![vec![i32::MAX; 3]; n];
    dp[0][0] = costs[0][0];
    dp[0][1] = costs[0][1];
    dp[0][2] = costs[0][2];
    for i in 1..n {
        dp[i][0] = dp[i - 1][1].min(dp[i - 1][2]) + costs[i][0];
        dp[i][1] = dp[i - 1][0].min(dp[i - 1][2]) + costs[i][1];
        dp[i][2] = dp[i - 1][0].min(dp[i - 1][1]) + costs[i][2];
    }
    dp[n - 1][0].min(dp[n - 1][1].min(dp[n - 1][2]))
}

pub fn check_x_matrix(grid: Vec<Vec<i32>>) -> bool {
    for i in 0..grid.len() {
        for j in 0..grid[0].len() {
            if i == j || i + j == grid.len() - 1 {
                if grid[i][j] == 0 {
                    return false;
                }
            } else if grid[i][j] != 0 {
                return false;
            }
        }
    }
    true
}

pub fn count_house_placements(n: i32) -> i32 {
    let mut dp: Vec<Vec<i64>> = vec![vec![0; 4]; n as usize];
    dp[0][0] = 1;
    dp[0][1] = 1;
    dp[0][2] = 1;
    dp[0][3] = 1;
    for i in 1..n as usize {
        dp[i][0] = (dp[i - 1][0] + dp[i - 1][1] + dp[i - 1][2] + dp[i - 1][3]) % 1_000_000_007;
        dp[i][1] = (dp[i - 1][0] + dp[i - 1][2]) % 1_000_000_007;
        dp[i][2] = (dp[i - 1][0] + dp[i - 1][1]) % 1_000_000_007;
        dp[i][3] = dp[i - 1][0];
    }
    let n = n as usize;
    ((dp[n - 1][0] + dp[n - 1][1] + dp[n - 1][2] + dp[n - 1][3]) % 1_000_000_007) as i32
}

pub fn find_lu_slength(strs: Vec<String>) -> i32 {
    fn is_substr(s1: &Vec<char>, s2: &Vec<char>) -> bool {
        let mut i1 = 0;
        let mut i2 = 0;
        let l1 = s1.len();
        let l2 = s2.len();
        while i1 < l1 && i2 < l2 {
            if s1[i1] == s2[i2] {
                i1 += 1;
            }
            i2 += 1;
        }
        if i1 == l1 {
            return true;
        }
        false
    }
    let strs = strs
        .into_iter()
        .map(|s| s.chars().collect::<Vec<char>>())
        .collect::<Vec<Vec<char>>>();
    let mut ans = -1;
    for i in 0..strs.len() {
        let mut flag = true;
        for j in 0..strs.len() {
            if i != j {
                if is_substr(&strs[i], &strs[j]) {
                    flag = false;
                    break;
                }
            }
        }
        if flag {
            ans = ans.max(strs[i].len() as i32);
        }
    }
    ans
}

pub fn wiggle_sort(nums: &mut Vec<i32>) {
    let x = (nums.len() + 1) / 2;
    let mut cpy = nums.clone();
    cpy.sort();
    let mut i = 0;
    let mut j = x - 1;
    let mut k = nums.len() - 1;
    while i < nums.len() {
        nums[i] = cpy[j];
        if i + 1 < nums.len() {
            nums[i + 1] = cpy[k]
        }
        i += 1;
        j -= 1;
        k -= 1;
    }
}

pub fn num_prime_arrangements(n: i32) -> i32 {
    fn prime_num(n: i32) -> i32 {
        // 第0位不要
        let mut nums = vec![1; n as usize + 1];
        for i in 2..nums.len() {
            if nums[i] == 1 {
                for j in 2..=n as usize / i {
                    nums[i * j] = 0;
                }
            }
        }
        nums.iter().skip(2).fold(0, |acc, x| acc + x)
    }
    let n1 = prime_num(n);
    let n2 = n - n1;
    let mut ans: i64 = 1;
    for i in 1..=n1 {
        ans = ans * i as i64 % 1_000_000_007;
    }
    for i in 1..=n2 {
        ans = ans * i as i64 % 1_000_000_007;
    }
    ans as i32
}

pub fn count_primes(n: i32) -> i32 {
    if n == 0 || n == 1 || n == 2 {
        return 0;
    }
    let mut nums = vec![1; n as usize];
    for num in 2..nums.len() {
        if nums[num] == 1 {
            for i in 2..=(n as usize - 1) / num {
                nums[num * i] = 0;
            }
        }
    }
    nums.iter().skip(2).fold(0, |acc, x| acc + x)
}

pub fn diff_ways_to_compute(expression: String) -> Vec<i32> {
    let mut ans = vec![];
    let n = expression.len();
    if n == 0 {
        return ans;
    }
    let c = expression.as_bytes();
    for i in 0..n {
        match c[i] {
            b'+' | b'-' | b'*' => {
                let left = diff_ways_to_compute(expression[..i].to_owned());
                let right = diff_ways_to_compute(expression[i + 1..].to_owned());
                for l in left.iter() {
                    for r in right.iter() {
                        match c[i] {
                            b'+' => ans.push(l + r),
                            b'-' => ans.push(l - r),
                            _ => ans.push(l * r),
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if ans.is_empty() {
        return vec![expression.parse::<i32>().unwrap()];
    }
    ans
}

pub fn min_refuel_stops(target: i32, start_fuel: i32, stations: Vec<Vec<i32>>) -> i32 {
    let mut dp = vec![0; stations.len() + 1];
    dp[0] = start_fuel;
    for (idx, station) in stations.iter().enumerate() {
        for j in (0..=idx).rev() {
            if dp[j] >= station[0] {
                dp[j + 1] = dp[j + 1].max(dp[j] + station[1]);
            }
        }
    }
    for i in 0..dp.len() {
        if dp[i] >= target {
            return i as i32;
        }
    }
    -1
}

pub fn decode_message(key: String, message: String) -> String {
    let mut hp = HashMap::new();
    let key = key.chars().collect::<Vec<char>>();
    for (_, &k) in key.iter().enumerate() {
        match hp.get(&k) {
            Some(_) => {
                continue;
            }
            None => {
                if k == ' ' {
                    continue;
                }
                hp.insert(k, ('a' as u8 + hp.len() as u8) as char);
            }
        }
    }
    let message = message.chars().collect::<Vec<char>>();
    let mut ans = Vec::<char>::new();
    for c in message {
        if c == ' ' {
            ans.push(' ');
            continue;
        }
        ans.push(hp.get(&c).unwrap().to_owned());
    }
    ans.iter().collect()
}

pub fn next_greater_element(n: i32) -> i32 {
    let mut nums = vec![];
    let ori = n;
    let mut n = n;
    while n != 0 {
        nums.push(n % 10);
        n /= 10;
    }
    let mut idx = -1;
    for i in 0..nums.len() {
        if i == nums.len() - 1 {
            return -1;
        } else if nums[i + 1] < nums[i] {
            idx = i as i32;
            break;
        }
    }
    let mut swap_idx = 0;
    while swap_idx <= idx {
        if nums[swap_idx as usize] > nums[idx as usize + 1] {
            break;
        }
        swap_idx += 1;
    }
    nums.swap(swap_idx as usize, idx as usize + 1);
    let mut cpy = nums.drain(0..=idx as usize).collect::<Vec<i32>>();
    cpy.sort_by(|a, b| b.cmp(a));
    cpy.append(&mut nums);
    cpy = cpy.into_iter().rev().collect::<Vec<i32>>();
    let mut ans: i64 = 0;
    for n in cpy {
        ans = ans * 10 + n as i64;
    }
    if ori > 0 && ans > i32::MAX as i64 {
        return -1;
    }
    ans as i32
}

pub fn minimum_abs_difference(arr: Vec<i32>) -> Vec<Vec<i32>> {
    let mut ans = vec![];
    let mut arr = arr;
    arr.sort();
    let mut min = i32::MAX;
    for i in 0..arr.len() - 1 {
        if arr[i + 1] - arr[i] < min {
            min = arr[i + 1] - arr[i];
            ans.drain(..);
            ans.push(vec![arr[i], arr[i + 1]]);
        } else if arr[i + 1] - arr[i] == min {
            ans.push(vec![arr[i], arr[i + 1]]);
        }
    }
    ans
}

pub fn replace_words(dictionary: Vec<String>, sentence: String) -> String {
    let sentence = sentence
        .split(" ")
        .map(|s| s.to_owned())
        .collect::<Vec<String>>();
    let mut ans: Vec<String> = vec![];
    for (i, s) in sentence.into_iter().enumerate() {
        let mut flag = false;
        for a in dictionary.iter() {
            if s.starts_with(a) {
                flag = true;
                if ans.len() >= i + 1 {
                    if ans[i].len() > a.len() {
                        ans[i] = a.to_owned();
                    }
                } else {
                    ans.push(a.to_owned());
                }
            }
        }
        if flag == false {
            ans.push(s);
        }
    }
    let mut ret = String::from("");
    let mut cnt = ans.len();
    for s in ans.into_iter() {
        ret.push_str(&s);
        cnt -= 1;
        if cnt != 0 {
            ret.push_str(" ");
        }
    }
    ret
}

pub fn min_cost_to_move_chips(position: Vec<i32>) -> i32 {
    position
        .iter()
        .fold((0, 0, 0), |acc, x| {
            (
                acc.0 + x % 2,
                acc.1 + (x + 1) % 2,
                (acc.0 + x % 2).min(acc.1 + (x + 1) % 2),
            )
        })
        .2
}

pub fn reverse_bits(x: u32) -> u32 {
    let mut ans = 0;
    let mut cnt = 32;
    let mut x = x;
    while cnt > 0 {
        cnt -= 1;
        ans = (ans << 1) + x % 2;
        x = x >> 1;
    }
    ans
}

pub fn is_isomorphic(s: String, t: String) -> bool {
    if s.len() != t.len() {
        return false;
    }
    let mut s2t = HashMap::new();
    let mut t2s = HashMap::new();
    let s = s.chars().collect::<Vec<char>>();
    let t = t.chars().collect::<Vec<char>>();
    for i in 0..s.len() {
        let ss = s2t.get(&s[i]);
        let tt = t2s.get(&t[i]);
        match (ss, tt) {
            (Some(ss), Some(tt)) => {
                if s[i] != *tt || t[i] != *ss {
                    return false;
                }
            }
            (Some(_), None) => {
                return false;
            }
            (None, Some(_)) => {
                return false;
            }
            (None, None) => {
                s2t.insert(s[i], t[i]);
                t2s.insert(t[i], s[i]);
            }
        }
    }
    true
}

pub fn is_power_of_two(n: i32) -> bool {
    if n >= 0 && (n & (n - 1)) == 0 {
        true
    } else {
        false
    }
}

pub fn is_ugly(n: i32) -> bool {
    if n <= 1 {
        return false;
    }
    let mut n = n;
    while n % 2 == 0 {
        n = n >> 1;
    }
    while n % 3 == 0 {
        n /= 3;
    }
    while n % 5 == 0 {
        n /= 5;
    }
    if n == 1 {
        return true;
    }
    false
}

pub fn missing_number(nums: Vec<i32>) -> i32 {
    let mut or_sum = 0;
    for i in 0..=nums.len() as i32 {
        or_sum ^= i;
    }
    for num in nums {
        or_sum ^= num;
    }
    or_sum
}

pub fn move_zeroes(nums: &mut Vec<i32>) {
    let mut j = 0;
    let mut idx = j;
    while j < nums.len() as i32 {
        if nums[j as usize] != 0 {
            j += 1;
        } else {
            let mut i = (j + 1).max(idx);
            while i < nums.len() as i32 && nums[i as usize] == 0 {
                i += 1;
            }
            if i == nums.len() as i32 {
                return;
            } else {
                nums.swap(i as usize, j as usize);
                idx = i + 1;
            }
            j += 1;
        }
    }
}

pub fn word_pattern(pattern: String, s: String) -> bool {
    let mut p2s = HashMap::new();
    let mut s2p = HashMap::new();
    let pattern = pattern.chars().collect::<Vec<char>>();
    let s = s.split(" ").map(|s| s.to_string()).collect::<Vec<String>>();
    if pattern.len() != s.len() {
        return false;
    }
    for i in 0..pattern.len() {
        let pp = p2s.get(&pattern[i]);
        let ss = s2p.get(&s[i]);
        match (pp, ss) {
            (None, None) => {
                p2s.insert(pattern[i], s[i].clone());
                s2p.insert(s[i].clone(), pattern[i]);
            }
            (Some(_), None) | (None, Some(_)) => {
                return false;
            }
            (Some(pp), Some(ss)) => {
                if *pp != s[i] || *ss != pattern[i] {
                    return false;
                }
            }
        }
    }
    true
}

pub fn is_power_of_three(n: i32) -> bool {
    if n <= 0 {
        return false;
    }
    let mut n = n;
    while n % 3 == 0 {
        n /= 3;
    }
    if n == 1 {
        return true;
    }
    false
}

pub fn len_longest_fib_subseq(arr: Vec<i32>) -> i32 {
    let mut hp = HashMap::<i32, usize>::new();
    for (i, n) in arr.iter().enumerate() {
        hp.insert(*n, i);
    }
    let mut ans = 0;
    let mut dp = vec![vec![0; arr.len()]; arr.len()];
    for i in 0..arr.len() {
        if i < 1 {
            continue;
        }
        for j in (0..=i - 1).rev() {
            if arr[j] * 2 <= arr[i] {
                break;
            }
            if let Some(&idx) = hp.get(&(arr[i] - arr[j])) {
                dp[j][i] = dp[j][i].max((dp[idx][j] + 1).max(3));
            }
            ans = ans.max(dp[j][i]);
        }
    }
    ans
}

pub fn cherry_pickup(grid: Vec<Vec<i32>>) -> i32 {
    let ans;
    let n = grid.len() as i32;
    let mut dp = vec![vec![vec![i32::MIN; n as usize]; n as usize]; 2 * n as usize - 1];
    dp[0][0][0] = grid[0][0];
    for k in 1..2 * n - 1 {
        for x1 in (k - n + 1).max(0)..=k.min(n - 1) {
            let y1 = k - x1;
            if grid[x1 as usize][y1 as usize] == -1 {
                continue;
            }
            for x2 in x1..=k.min(n - 1) {
                let y2 = k - x2;
                if grid[x2 as usize][y2 as usize] == -1 {
                    continue;
                }
                let mut res = dp[k as usize - 1][x1 as usize][x2 as usize];
                if x1 != 0 {
                    res = res.max(dp[k as usize - 1][x1 as usize - 1][x2 as usize]);
                }
                if x2 != 0 {
                    res = res.max(dp[k as usize - 1][x1 as usize][x2 as usize - 1]);
                }
                if x1 != 0 && x2 != 0 {
                    res = res.max(dp[k as usize - 1][x1 as usize - 1][x2 as usize - 1]);
                }
                res += grid[x1 as usize][y1 as usize];
                if x1 != x2 {
                    res += grid[x2 as usize][y2 as usize];
                }
                dp[k as usize][x1 as usize][x2 as usize] = res;
            }
        }
    }
    ans = dp[2 * n as usize - 2][n as usize - 1][n as usize - 1].max(0);
    ans
}

pub fn odd_cells(m: i32, n: i32, indices: Vec<Vec<i32>>) -> i32 {
    let mut array = vec![vec![0; n as usize]; m as usize];
    let mut ans = 0;
    for index in indices {
        for j in 0..n as usize {
            array[index[0] as usize][j] += 1;
        }
        for i in 0..m as usize {
            array[i][index[1] as usize] += 1;
        }
    }
    for i in 0..m as usize {
        for j in 0..n as usize {
            if array[i][j] % 2 != 0 {
                ans += 1;
            }
        }
    }
    ans
}

pub fn compute_area(
    ax1: i32,
    ay1: i32,
    ax2: i32,
    ay2: i32,
    bx1: i32,
    by1: i32,
    bx2: i32,
    by2: i32,
) -> i32 {
    let area1 = (ax2 - ax1) * (ay2 - ay1);
    let area2 = (bx2 - bx1) * (by2 - by1);
    if ax2 <= bx1 || ax1 >= bx2 || ay2 <= by1 || ay1 >= by2 {
        return area1 + area2;
    }
    let mut x = vec![ax1, ax2, bx1, bx2];
    let mut y = vec![ay1, ay2, by1, by2];
    x.sort();
    y.sort();
    area1 + area2 - (x[2] - x[1]) * (y[2] - y[1])
}

pub fn num_squares(n: i32) -> i32 {
    fn is_square(n: i32) -> bool {
        let nr = (n as f64).sqrt() as i32;
        return nr * nr == n;
    }
    fn is4square(n: i32) -> bool {
        let mut n = n;
        while n % 4 == 0 {
            n /= 4;
        }
        return n % 8 == 7;
    }
    if is_square(n) {
        return 1;
    } else if is4square(n) {
        return 4;
    }
    for i in 1..=(n as f64).sqrt() as i32 {
        let j = n - i * i;
        if is_square(j) {
            return 2;
        }
    }
    return 3;
}

pub fn asteroid_collision(asteroids: Vec<i32>) -> Vec<i32> {
    let mut st = vec![];
    for asteroid in asteroids {
        if asteroid < 0 {
            if st.is_empty() {
                st.push(asteroid);
                continue;
            }
            while let Some(a) = st.pop() {
                if a < 0 {
                    st.push(a);
                    st.push(asteroid);
                    break;
                } else if a < asteroid.abs() {
                    if st.is_empty() {
                        st.push(asteroid);
                        break;
                    }
                    continue;
                } else if a > asteroid.abs() {
                    st.push(a);
                    break;
                } else {
                    break;
                }
            }
        } else {
            st.push(asteroid);
        }
    }
    st
}

pub fn get_money_amount(n: i32) -> i32 {
    let mut dp = vec![vec![0; n as usize + 1]; n as usize + 1];
    for i in (1..=n as usize - 1).rev() {
        for j in i + 1..=n as usize {
            dp[i][j] = j as i32 + dp[i][j - 1];
            for k in i..j {
                dp[i][j] = dp[i][j].min(k as i32 + dp[i][k - 1].max(dp[k + 1][j]));
            }
        }
    }
    dp[1][n as usize]
}

pub fn predict_the_winner(nums: Vec<i32>) -> bool {
    let mut dp = vec![vec![0; nums.len()]; nums.len()];
    for i in 0..nums.len() {
        dp[i][i] = nums[i];
    }
    for i in (0..nums.len()).rev() {
        for j in i + 1..nums.len() {
            dp[i][j] = (nums[i] - dp[i + 1][j]).max(nums[j] - dp[i][j - 1]);
        }
    }
    if dp[0][nums.len() - 1] >= 0 {
        return true;
    }
    false
}

pub fn stone_game(piles: Vec<i32>) -> bool {
    let mut dp = vec![vec![0; piles.len()]; piles.len()];
    for i in 0..piles.len() {
        dp[i][i] = piles[i];
    }
    for i in (0..piles.len()).rev() {
        for j in i + 1..piles.len() {
            dp[i][j] = (piles[i] - dp[i + 1][j]).max(piles[j] - dp[i][j - 1]);
        }
    }
    if dp[0][piles.len() - 1] > 0 {
        return true;
    }
    false
}

pub fn stone_game_iii(stone_value: Vec<i32>) -> String {
    let n = stone_value.len();
    let mut dp = vec![0; n + 1];
    dp[n - 1] = stone_value[n - 1];
    dp[n] = 0;
    for i in (0..n - 1).rev() {
        dp[i] = stone_value[i] - dp[i + 1];
        if n - i >= 2 {
            dp[i] = dp[i].max(stone_value[i] + stone_value[i + 1] - dp[i + 2]);
        }
        if n - i >= 3 {
            dp[i] = dp[i].max(stone_value[i] + stone_value[i + 1] + stone_value[i + 2] - dp[i + 3]);
        }
    }
    if dp[0] > 0 {
        return "Alice".to_owned();
    } else if dp[0] == 0 {
        return "Tie".to_owned();
    } else {
        return "Bob".to_owned();
    }
}

pub fn winner_square_game(n: i32) -> bool {
    let n = n as usize;
    let mut dp = vec![false; n + 2];
    dp[n] = true;
    dp[n + 1] = false;
    for i in (1..n).rev() {
        let mut j = 1;
        loop {
            if j * j <= n - i + 1 {
                dp[i] = dp[i] || !dp[i + j * j];
                if dp[i] == true {
                    break;
                }
            } else {
                break;
            }
            j += 1;
        }
    }
    dp[1]
}

pub fn max_coins(piles: Vec<i32>) -> i32 {
    let mut piles = piles;
    piles.sort();
    let mut i = 0;
    let mut ans = 0;
    let mut j = piles.len() - 1;
    while i < j {
        ans += piles[j - 1];
        i += 1;
        j -= 2;
    }
    ans
}
pub fn stone_game_vi(alice_values: Vec<i32>, bob_values: Vec<i32>) -> i32 {
    let n = alice_values.len();
    let mut total = vec![(0, 0); n];
    for i in 0..n {
        total[i].0 = alice_values[i] + bob_values[i];
        total[i].1 = i;
    }
    total.sort_by(|a, b| b.0.cmp(&a.0));
    let mut sum1 = 0;
    let mut sum2 = 0;
    for i in 0..n {
        if i % 2 == 0 {
            sum1 += alice_values[total[i].1];
        } else {
            sum2 += bob_values[total[i].1];
        }
    }
    if sum1 > sum2 {
        return 1;
    } else if sum1 == sum2 {
        return 0;
    } else {
        return -1;
    }
}

pub fn repeated_string_match(a: String, b: String) -> i32 {
    let lena = a.len();
    let lenb = b.len();
    let n = 3 + lenb / lena;
    let mut s = a.clone();
    for i in 1..=n {
        if s.contains(&b.clone()) {
            return i as i32;
        }
        s = s + &a.clone();
    }
    -1
}

pub fn camel_match(queries: Vec<String>, pattern: String) -> Vec<bool> {
    let mut ans = vec![];
    let pattern = pattern.chars().collect::<Vec<char>>();
    let n = pattern.len();
    for query in queries {
        let q = query.as_bytes();
        let (mut i, mut j) = (0, 0);
        let mut flag = true;
        while i < n && j < q.len() {
            if q[j] != pattern[i] as u8 {
                if q[j].is_ascii_lowercase() {
                    j += 1;
                    continue;
                } else {
                    ans.push(false);
                    flag = false;
                    break;
                }
            } else if q[j] == pattern[i] as u8 {
                i += 1;
                j += 1;
                continue;
            }
        }
        if flag == false {
            continue;
        } else {
            if i == n {
                while j < q.len() {
                    if !q[j].is_ascii_lowercase() {
                        ans.push(false);
                        flag = false;
                        break;
                    }
                    j += 1;
                }
                if flag == false {
                    continue;
                } else {
                    ans.push(true);
                }
            } else {
                ans.push(false);
            }
        }
    }
    ans
}

pub fn string_matching(words: Vec<String>) -> Vec<String> {
    let mut ans = vec![];
    let mut words = words;
    words.sort_by(|a, b| a.len().cmp(&b.len()));
    for i in 0..words.len() {
        for j in i + 1..words.len() {
            if words[j].contains(&words[i]) {
                ans.push(words[i].clone());
                break;
            }
        }
    }
    ans
}

pub fn is_prefix_of_word(sentence: String, search_word: String) -> i32 {
    let sentence = sentence.split(" ").collect::<Vec<_>>();
    for (i, s) in sentence.iter().enumerate() {
        if s.strip_prefix(&search_word).is_some() {
            return i as i32 + 1;
        }
    }
    -1
}

pub fn max_repeating(sequence: String, word: String) -> i32 {
    let n = sequence.len() / word.len();
    let mut s = word.clone();
    let mut ret = 0;
    for i in 1..=n {
        if sequence.contains(&s) {
            ret = i as i32;
        }
        s = s + &word.clone();
    }
    ret
}

pub fn is_fliped_string(s1: String, s2: String) -> bool {
    if s1.len() != s2.len() {
        return false;
    } else if s1.is_empty() {
        return true;
    } else if s1 == s2 {
        return true;
    }
    let s = s1.clone() + &s1;
    if s.contains(&s2) {
        return true;
    }
    false
}

pub fn multi_search(big: String, smalls: Vec<String>) -> Vec<Vec<i32>> {
    let mut ans = vec![];
    for small in smalls {
        if small.is_empty() {
            ans.push(vec![]);
            continue;
        }
        let mut temp = vec![];
        let mut big = big.clone();
        let mut pre = 0;
        while let Some(idx) = big.find(&small) {
            temp.push(idx as i32 + pre as i32);
            let mut itr = big.chars().collect::<Vec<_>>();
            itr.drain(0..=idx);
            big = itr.iter().collect();
            pre += idx + 1;
        }
        ans.push(temp);
    }
    ans
}

pub fn can_choose(groups: Vec<Vec<i32>>, nums: Vec<i32>) -> bool {
    let mut anchor = 0;
    for group in groups {
        let mut flag = false;
        let mut t = 0;
        for idx in anchor..nums.len() {
            if nums[idx] == group[0] {
                flag = true;
                let len = group.len();
                if nums.len() - idx < len {
                    return false;
                } else {
                    for k in idx..idx + len {
                        if nums[k] != group[k - idx] {
                            flag = false;
                            break;
                        }
                    }
                    if flag {
                        t = idx;
                        break;
                    }
                }
            }
        }
        if flag {
            anchor = t + group.len()
        } else {
            return false;
        }
    }
    true
}

pub fn shift_grid(grid: Vec<Vec<i32>>, k: i32) -> Vec<Vec<i32>> {
    let mut ans = vec![vec![0; grid[0].len()]; grid.len()];
    let m = grid.len();
    let n = grid[0].len();
    let k = k as usize % (m * n);
    for i in 0..m {
        for j in 0..n {
            let ni = (i + (j + k) / n) % m;
            let nj = (j + k) % n;
            ans[ni][nj] = grid[i][j];
        }
    }
    ans
}

pub fn permutation(s: String) -> Vec<String> {
    let mut ans = vec![];
    let s = s.chars().collect::<Vec<_>>();
    let mut map = vec![false; s.len()];
    let n = s.len();
    fn dfs(
        s: &Vec<char>,
        map: &mut Vec<bool>,
        n: usize,
        ans: &mut Vec<String>,
        tmp: &mut Vec<char>,
    ) {
        if tmp.len() == n {
            ans.push(tmp.iter().collect());
            return;
        } else {
            for i in 0..n {
                if map[i] == false {
                    map[i] = true;
                    tmp.push(s[i]);
                    dfs(s, map, n, ans, tmp);
                    tmp.pop();
                    map[i] = false;
                }
            }
        }
    }
    let mut tmp = vec![];
    dfs(&s, &mut map, n, &mut ans, &mut tmp);
    ans
}

pub fn max_ice_cream(costs: Vec<i32>, coins: i32) -> i32 {
    let mut costs = costs;
    costs.sort();
    let mut ans = 0;
    let mut coins = coins;
    for i in 0..costs.len() {
        if coins >= costs[i] {
            ans += 1;
            coins -= costs[i];
        } else {
            break;
        }
    }
    ans
}

pub fn two_city_sched_cost(costs: Vec<Vec<i32>>) -> i32 {
    let mut diff = costs.iter().map(|c| c[0] - c[1]).collect::<Vec<_>>();
    let mut sumb = costs.iter().fold(0, |acc, x| acc + x[1]);
    diff.sort();
    for i in 0..costs.len() / 2 {
        sumb += diff[i];
    }
    sumb
}

pub fn min_cost_1547(n: i32, cuts: Vec<i32>) -> i32 {
    let mut idx = vec![0; cuts.len() + 2];
    let mut cuts = cuts;
    cuts.sort();
    let m = cuts.len();
    for i in 1..=cuts.len() {
        idx[i] = cuts[i - 1];
    }
    idx[m + 1] = n;
    let mut dp = vec![vec![0; m + 2]; m + 2];
    for i in (1..=m).rev() {
        for j in i..=m {
            dp[i][j] = dp[i][i - 1] + dp[i + 1][j];
            for k in i + 1..=j {
                dp[i][j] = dp[i][j].min(dp[i][k - 1] + dp[k + 1][j]);
            }
            dp[i][j] += idx[j + 1] - idx[i - 1];
        }
    }
    dp[1][m]
}

pub fn get_last_moment(n: i32, left: Vec<i32>, right: Vec<i32>) -> i32 {
    left.iter()
        .fold(0, |acc, x| acc.max(x - 0))
        .max(right.iter().fold(0, |acc, x| acc.max(n - x)))
}

pub fn num_of_pairs(nums: Vec<String>, target: String) -> i32 {
    let mut ans = 0;
    for i in 0..nums.len() {
        for j in 0..nums.len() {
            if i == j {
                continue;
            }
            if nums[i].clone() + &nums[j] == target {
                ans += 1;
            }
        }
    }
    ans
}

pub fn sum_zero(n: i32) -> Vec<i32> {
    let mut ans = vec![];
    if n % 2 == 1 {
        ans.push(0);
    }
    for i in 1..=n / 2 {
        ans.push(i);
        ans.push(-i);
    }
    ans
}

pub fn max_chunks_to_sorted(arr: Vec<i32>) -> i32 {
    if arr.len() == 1 {
        return 1;
    }
    let mut ans = 0;
    let mut bound = arr[0];
    if bound == arr.len() as i32 - 1 {
        return 1;
    }
    for (i, &num) in arr.iter().enumerate().skip(1) {
        if i as i32 > bound {
            ans += 1;
            bound = num;
            if bound == arr.len() as i32 - 1 {
                ans += 1;
                break;
            }
        } else {
            if num > bound {
                bound = num;
                if bound == arr.len() as i32 - 1 {
                    ans += 1;
                    break;
                }
            }
        }
    }
    ans
}

pub fn flip_and_invert_image(image: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut ans = vec![];
    for i in 0..image.len() {
        ans.push(
            image[i]
                .clone()
                .into_iter()
                .map(|x| 1 - x)
                .rev()
                .collect::<Vec<_>>(),
        );
    }
    ans
}

pub fn validate_stack_sequences(pushed: Vec<i32>, popped: Vec<i32>) -> bool {
    let mut st = vec![];
    let mut anchor = 0;
    for i in 0..pushed.len() {
        st.push(pushed[i]);
        while !st.is_empty() && *st.last().unwrap() == popped[anchor] {
            st.pop();
            anchor += 1;
        }
    }
    if anchor == popped.len() {
        return true;
    } else {
        while !st.is_empty() {
            if *st.last().unwrap() == popped[anchor] {
                st.pop();
                anchor += 1;
            } else {
                return false;
            }
        }
        true
    }
}

pub fn sequence_reconstruction(nums: Vec<i32>, sequences: Vec<Vec<i32>>) -> bool {
    let n = nums.len();
    let mut indegree = vec![0; n];
    let mut hp = HashMap::new();
    for s in sequences {
        for i in 1..s.len() {
            indegree[s[i] as usize - 1] += 1;
            let e = hp.entry(s[i - 1]).or_insert(vec![]);
            e.push(s[i]);
        }
    }
    let mut st = vec![];
    for i in indegree.iter() {
        if *i == 0 {
            st.push(*i + 1);
        }
    }
    if st.len() > 1 {
        return false;
    }
    while !st.is_empty() {
        let s = st.pop().unwrap();
        let v = hp.get(&s);
        if v.is_none() {
            break;
        }
        let v = v.unwrap().clone();
        for node in v {
            indegree[node as usize - 1] -= 1;
            if indegree[node as usize - 1] == 0 {
                st.push(node);
            }
        }
        if st.len() > 1 {
            return false;
        }
    }
    true
}

pub fn distance_between_bus_stops(distance: Vec<i32>, start: i32, destination: i32) -> i32 {
    let total = distance.iter().sum::<i32>();
    let min = start.min(destination);
    let max = start.max(destination);
    let mut ans = 0;
    for i in min..max {
        ans += distance[i as usize];
    }
    ans.min(total - ans)
}

pub fn array_rank_transform(arr: Vec<i32>) -> Vec<i32> {
    let mut ans = vec![0; arr.len()];
    let mut arr = arr
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i as i32, x))
        .collect::<Vec<(i32, i32)>>();
    arr.sort_by_key(|(_, x)| *x);
    let mut anchor = 1;
    for i in 0..arr.len() {
        if i == 0 {
            ans[arr[0].0 as usize] = anchor;
            anchor += 1;
        } else {
            if arr[i].1 == arr[i - 1].1 {
                ans[arr[i].0 as usize] = ans[arr[i - 1].0 as usize];
            } else {
                ans[arr[i].0 as usize] = anchor;
                anchor += 1;
            }
        }
    }
    ans
}

pub fn valid_square(p1: Vec<i32>, p2: Vec<i32>, p3: Vec<i32>, p4: Vec<i32>) -> bool {
    let mut nodes = vec![p1, p2, p3, p4];
    nodes.sort_by(|a, b| {
        if a[0] != b[0] {
            a[0].cmp(&b[0])
        } else {
            a[1].cmp(&b[1])
        }
    });
    let (x1, y1) = (nodes[0][0], nodes[0][1]);
    let (x2, y2) = (nodes[1][0], nodes[1][1]);
    let (x3, y3) = (nodes[2][0], nodes[2][1]);
    let (x4, y4) = (nodes[3][0], nodes[3][1]);
    if x1 == x2 && x1 == x3 && x3 == x4 {
        return false;
    }
    if x1 == x2
        && x3 == x4
        && y1 == y3
        && y2 == y4
        && (x2 - x1).pow(2) + (y2 - y1).pow(2) == (x3 - x1).pow(2) + (y3 - y1).pow(2)
    {
        return true;
    } else if (x2 - x1) * (y4 - y3) == (x4 - x3) * (y2 - y1)
        && (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
            == (x4 - x3) * (x4 - x3) + (y4 - y3) * (y4 - y3)
        && (x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1) == 0
        && (x2 - x1).pow(2) + (y2 - y1).pow(2) == (x3 - x1).pow(2) + (y3 - y1).pow(2)
    {
        return true;
    }
    false
}

pub fn find_kth_largest(nums: Vec<i32>, k: i32) -> i32 {
    let mut nums = nums;
    fn random_partition(nums: &mut Vec<i32>, l: i32, r: i32) -> i32 {
        let idx = rand::thread_rng().gen_range(l..r + 1);
        let x = nums[idx as usize];
        let t = nums[r as usize];
        nums[r as usize] = x;
        nums[idx as usize] = t;
        let mut p1 = l - 1;
        let mut p2 = l;
        while p2 <= r {
            if nums[p2 as usize] <= x {
                p1 += 1;
                let t = nums[p2 as usize];
                nums[p2 as usize] = nums[p1 as usize];
                nums[p1 as usize] = t;
            }
            p2 += 1;
        }
        p1
    }
    fn select(nums: &mut Vec<i32>, l: i32, r: i32, k: i32, n: i32) -> i32 {
        let mid = random_partition(nums, l, r);
        if mid == n - k {
            return nums[mid as usize];
        } else if mid < n - k {
            return select(nums, mid + 1, r, k, n);
        } else {
            return select(nums, l, mid - 1, k, n);
        }
    }
    let n = nums.len() as i32;
    return select(&mut nums, 0, n - 1, k, n);
}

pub fn top_k_frequent(nums: Vec<i32>, k: i32) -> Vec<i32> {
    let mut ans = vec![];
    let mut hp = HashMap::new();
    for n in nums {
        let e = hp.entry(n).or_insert(0);
        *e += 1;
    }
    let mut tmp = vec![];
    for (k, v) in hp.iter() {
        tmp.push((*k, *v));
    }
    tmp.sort_by_key(|x| x.1);
    for t in tmp.iter().rev() {
        ans.push(t.0);
        if ans.len() == k as usize {
            break;
        }
    }
    ans
}

pub fn smallest_k(arr: Vec<i32>, k: i32) -> Vec<i32> {
    let mut nums = arr;
    fn random_partition(nums: &mut Vec<i32>, l: i32, r: i32) -> i32 {
        let idx = rand::thread_rng().gen_range(l..r + 1);
        let x = nums[idx as usize];
        let t = nums[r as usize];
        nums[r as usize] = x;
        nums[idx as usize] = t;
        let mut p1 = l - 1;
        let mut p2 = l;
        while p2 <= r {
            if nums[p2 as usize] <= x {
                p1 += 1;
                let t = nums[p2 as usize];
                nums[p2 as usize] = nums[p1 as usize];
                nums[p1 as usize] = t;
            }
            p2 += 1;
        }
        p1
    }
    fn select(nums: &mut Vec<i32>, l: i32, r: i32, k: i32) -> Vec<i32> {
        let mid = random_partition(nums, l, r);
        if mid == k - 1 {
            return nums[0..=mid as usize].to_vec();
        } else if mid < k - 1 {
            return select(nums, mid + 1, r, k);
        } else {
            return select(nums, l, mid - 1, k);
        }
    }
    let n = nums.len();
    select(&mut nums, 0, n as i32 - 1, k)
}

pub fn kth_largest_value(matrix: Vec<Vec<i32>>, k: i32) -> i32 {
    let mut pre = vec![vec![0; matrix[0].len()]; matrix.len()];
    let mut nums = vec![];
    let m = matrix.len();
    let n = matrix[0].len();
    for i in 0..m {
        for j in 0..n {
            if j > 0 {
                pre[i][j] ^= pre[i][j - 1];
            }
            if i > 0 {
                pre[i][j] ^= pre[i - 1][j];
            }
            if i > 0 && j > 0 {
                pre[i][j] ^= pre[i - 1][j - 1];
            }
            pre[i][j] ^= matrix[i][j];
            nums.push(pre[i][j]);
        }
    }
    nums.sort();
    nums[nums.len() - k as usize]
}

pub fn kth_largest_number(nums: Vec<String>, k: i32) -> String {
    let mut nums = nums;
    nums.sort_by(|a, b| {
        if a.len() != b.len() {
            a.len().cmp(&b.len())
        } else {
            a.cmp(b)
        }
    });
    nums[nums.len() - k as usize].clone()
}

pub fn generate_the_string(n: i32) -> String {
    let mut ans;
    if n % 2 == 1 {
        ans = vec!['a'; n as usize].iter().collect();
    } else {
        ans = vec!['a'; n as usize - 1].iter().collect();
        ans = ans + "b";
    }
    ans
}

pub fn min_subsequence(nums: Vec<i32>) -> Vec<i32> {
    let mut nums = nums;
    nums.sort_by(|a, b| b.cmp(&a));
    let mut ans = vec![];
    let sum = nums.iter().sum::<i32>();
    let mut tmp = 0;
    for i in 0..nums.len() {
        ans.push(nums[i]);
        tmp += nums[i];
        if tmp > sum - tmp {
            break;
        }
    }
    ans
}

pub fn xor_game(nums: Vec<i32>) -> bool {
    if nums.len() % 2 == 0 {
        return true;
    } else {
        let mut xor = 0;
        for num in nums {
            xor ^= num;
        }
        if xor == 0 {
            return true;
        }
    }
    false
}

pub fn min_start_value(nums: Vec<i32>) -> i32 {
    let mut min = nums[0];
    min = nums
        .iter()
        .skip(1)
        .fold((min, min), |acc, x| (x + acc.0, acc.1.min(x + acc.0)))
        .1;
    if min >= 0 {
        return 1;
    }
    -min + 1
}

pub fn group_anagrams(strs: Vec<String>) -> Vec<Vec<String>> {
    let mut hp = HashMap::new();
    for s in strs {
        let mut ss = s.clone().chars().collect::<Vec<_>>();
        ss.sort();
        let ss = ss.iter().collect::<String>();
        let e = hp.entry(ss).or_insert(vec![]);
        e.push(s);
    }
    let mut ans = vec![];
    for (_, v) in hp.iter() {
        ans.push(v.clone());
    }
    ans
}

pub fn distribute_candies(candy_type: Vec<i32>) -> i32 {
    let mut candy_type = candy_type;
    candy_type.sort();
    let n = candy_type.len() / 2;
    let mut ans = 1;
    for i in 1..candy_type.len() {
        if ans == n as i32 {
            return n as i32;
        }
        if candy_type[i] != candy_type[i - 1] {
            ans += 1;
        }
    }
    ans
}

pub fn find_lhs(nums: Vec<i32>) -> i32 {
    let mut nums = nums;
    nums.sort();
    let mut ans = 0;
    let mut begin = 0;
    let mut end = 0;
    loop {
        while end < nums.len() - 1 && nums[end] == nums[end + 1] {
            end += 1;
        }
        end += 1;
        if end == nums.len() {
            break;
        }
        let tmp = end;
        if nums[end] - nums[begin] == 1 {
            while end < nums.len() - 1 && nums[end] == nums[end + 1] {
                end += 1;
            }
            ans = ans.max(end as i32 - begin as i32 + 1);
        }
        begin = tmp;
    }
    ans
}

pub fn matrix_reshape(mat: Vec<Vec<i32>>, r: i32, c: i32) -> Vec<Vec<i32>> {
    if r * c != mat[0].len() as i32 * mat.len() as i32 {
        return mat;
    }
    let mut ans = vec![vec![0; c as usize]; r as usize];
    let mut anchor_x = 0;
    let mut anchor_y = 0;
    for i in 0..r as usize {
        for j in 0..c as usize {
            ans[i][j] = mat[anchor_x][anchor_y];
            anchor_y += 1;
            if anchor_y == mat[0].len() {
                anchor_x += 1;
                anchor_y = 0;
            }
        }
    }
    ans
}

pub fn find_unsorted_subarray(nums: Vec<i32>) -> i32 {
    let mut minn = i32::MAX;
    let mut maxn = i32::MIN;
    let mut left = -1;
    let mut right = -1;
    for i in 0..nums.len() {
        if nums[i] >= maxn {
            maxn = nums[i];
        } else {
            right = i as i32;
        }
        if nums[nums.len() - i - 1] <= minn {
            minn = nums[nums.len() - i - 1];
        } else {
            left = nums.len() as i32 - i as i32 - 1;
        }
    }
    if right == -1 {
        return 0;
    }
    right - left + 1
}

pub fn are_numbers_ascending(s: String) -> bool {
    let s = s.split(" ").collect::<Vec<_>>();
    let mut pre = 0;
    for i in 0..s.len() {
        let token = s[i].to_owned().chars().collect::<Vec<_>>();
        if token[0] as u8 >= '1' as u8 && token[0] as u8 <= '9' as u8 {
            let num = token.iter().collect::<String>().parse::<i32>().unwrap();
            if num > pre {
                pre = num;
            } else {
                return false;
            }
        }
    }
    true
}

pub fn smallest_number(num: i64) -> i64 {
    if num == 0 {
        return 0;
    }
    let mut flag = false;
    if num < 0 {
        flag = true;
    }
    let mut nums = vec![];
    let mut num = num.abs();
    while num != 0 {
        nums.push(num % 10);
        num /= 10;
    }
    if flag {
        nums.sort_by(|a, b| b.cmp(&a));
    } else {
        nums.sort();
        if nums[0] == 0 {
            for i in 0..nums.len() {
                if nums[i] != 0 {
                    nums.swap(0, i);
                    break;
                }
            }
        }
    }
    let mut ans = 0;
    nums.iter().for_each(|x| {
        ans = ans * 10 + *x;
    });
    if flag {
        -ans
    } else {
        ans
    }
}

pub fn max_count(m: i32, n: i32, ops: Vec<Vec<i32>>) -> i32 {
    let mut row_cnt = vec![0; m as usize];
    let mut col_cnt = vec![0; n as usize];
    for op in ops {
        for i in 0..op[0] as usize {
            row_cnt[i] += 1;
        }
        for j in 0..op[1] as usize {
            col_cnt[j] += 1;
        }
    }
    let mut max = i32::MIN;
    let mut max_cnt1 = 0;
    for i in 0..row_cnt.len() {
        if row_cnt[i] > max {
            max = row_cnt[i];
            max_cnt1 = 1;
        } else if row_cnt[i] == max {
            max_cnt1 += 1;
        }
    }
    let mut max = i32::MIN;
    let mut max_cnt2 = 0;
    for i in 0..col_cnt.len() {
        if col_cnt[i] > max {
            max = col_cnt[i];
            max_cnt2 = 1;
        } else if col_cnt[i] == max {
            max_cnt2 += 1;
        }
    }
    max_cnt1 * max_cnt2
}

pub fn find_duplicate(paths: Vec<String>) -> Vec<Vec<String>> {
    let mut hp = HashMap::new();
    let mut ans = vec![];
    for path in paths {
        let path_mut = path.split(" ").collect::<Vec<_>>();
        let dir = path_mut[0].to_owned();
        for i in 1..path_mut.len() {
            let left = path_mut[i].find("(").unwrap();
            let subpath = path_mut[i].to_owned().chars().collect::<Vec<_>>();
            let filename = subpath[0..left].iter().collect::<String>();
            let content = subpath[left + 1..].iter().collect::<String>();
            let e = hp.entry(content).or_insert(vec![]);
            e.push(dir.clone() + "/" + &filename);
        }
    }
    for (_, v) in hp {
        if v.len() >= 2 {
            ans.push(v);
        }
    }
    ans
}

pub fn solve_equation(equation: String) -> String {
    let pos = equation.find("=").unwrap();
    let equationl = equation[0..pos].to_owned().chars().collect::<Vec<_>>();
    let equationr = equation[pos + 1..].to_owned().chars().collect::<Vec<_>>();
    fn parse_equation(equation: &Vec<char>) -> (i32, i32) {
        let mut x = 0;
        let mut a = 0;
        let mut i = 0;
        loop {
            if i == equation.len() {
                break;
            }
            let mut flag = true;
            if i != 0 || equation[i] == '-' || equation[i] == '+' {
                if equation[i] == '-' {
                    flag = false;
                }
                i += 1;
            }
            if equation[i] == 'x' {
                i += 1;
                match flag {
                    true => x += 1,
                    false => x -= 1,
                };
                continue;
            } else {
                let mut j = i;
                while j < equation.len()
                    && equation[j] != 'x'
                    && equation[j] != '+'
                    && equation[j] != '-'
                {
                    j += 1;
                }
                if j == equation.len() {
                    if flag {
                        a += equation[i..j]
                            .iter()
                            .collect::<String>()
                            .parse::<i32>()
                            .unwrap();
                    } else {
                        a -= equation[i..j]
                            .iter()
                            .collect::<String>()
                            .parse::<i32>()
                            .unwrap();
                    }
                } else if equation[j] == 'x' {
                    if flag {
                        x += equation[i..j]
                            .iter()
                            .collect::<String>()
                            .parse::<i32>()
                            .unwrap();
                    } else {
                        x -= equation[i..j]
                            .iter()
                            .collect::<String>()
                            .parse::<i32>()
                            .unwrap();
                    }
                    j += 1;
                } else {
                    if flag {
                        a += equation[i..j]
                            .iter()
                            .collect::<String>()
                            .parse::<i32>()
                            .unwrap();
                    } else {
                        a -= equation[i..j]
                            .iter()
                            .collect::<String>()
                            .parse::<i32>()
                            .unwrap();
                    }
                }
                i = j;
            }
        }
        (x, a)
    }
    let (left_x, left_a) = parse_equation(&equationl);
    let (right_x, right_a) = parse_equation(&equationr);
    let x = left_x - right_x;
    let a = right_a - left_a;
    if x == 0 && a != 0 {
        return "No solution".to_owned();
    } else if x == 0 && a == 0 {
        return "Infinite solutions".to_owned();
    }
    format!("x={}", a / x)
}

pub fn are_occurrences_equal(s: String) -> bool {
    let mut map = [0; 26];
    s.chars().for_each(|c| map[c as usize - 'a' as usize] += 1);
    let mut pre = -1;
    for i in 0..26 {
        if map[i] != 0 && pre == -1 {
            pre = map[i];
        } else if map[i] != 0 && map[i] != pre {
            return false;
        }
    }
    true
}

pub fn nearest_exit(maze: Vec<Vec<char>>, entrance: Vec<i32>) -> i32 {
    let m = maze.len();
    let n = maze[0].len();
    let mut vis = vec![vec![0; n]; m];
    let mut poses = VecDeque::new();
    poses.push_back((entrance[0], entrance[1], 0));
    vis[entrance[0] as usize][entrance[1] as usize] = 1;
    while !poses.is_empty() {
        let node = poses.pop_front().unwrap();
        if node.2 != 0
            && (node.0 == 0 || node.0 == m as i32 - 1 || node.1 == 0 || node.1 == n as i32 - 1)
        {
            return node.2;
        }
        if node.0 != 0
            && vis[node.0 as usize - 1][node.1 as usize] != 1
            && maze[node.0 as usize - 1][node.1 as usize] != '+'
        {
            poses.push_back((node.0 - 1, node.1, node.2 + 1));
            vis[node.0 as usize - 1][node.1 as usize] = 1
        }
        if node.0 != m as i32 - 1
            && vis[node.0 as usize + 1][node.1 as usize] != 1
            && maze[node.0 as usize + 1][node.1 as usize] != '+'
        {
            poses.push_back((node.0 + 1, node.1, node.2 + 1));
            vis[node.0 as usize + 1][node.1 as usize] = 1
        }
        if node.1 != 0
            && vis[node.0 as usize][node.1 as usize - 1] != 1
            && maze[node.0 as usize][node.1 as usize - 1] != '+'
        {
            poses.push_back((node.0, node.1 - 1, node.2 + 1));
            vis[node.0 as usize][node.1 as usize - 1] = 1
        }
        if node.1 != n as i32 - 1
            && vis[node.0 as usize][node.1 as usize + 1] != 1
            && maze[node.0 as usize][node.1 as usize + 1] != '+'
        {
            poses.push_back((node.0, node.1 + 1, node.2 + 1));
            vis[node.0 as usize][node.1 as usize + 1] = 1
        }
    }
    -1
}

pub fn min_set_size(arr: Vec<i32>) -> i32 {
    let mut hp = HashMap::new();
    let total = arr.len() as i32;
    for n in arr {
        let e = hp.entry(n).or_insert(0);
        *e += 1;
    }
    let mut nums = hp.into_iter().collect::<Vec<_>>();
    nums.sort_by(|a, b| b.1.cmp(&a.1));
    let mut tmp = total as i32;
    let mut ans = 0;
    while tmp > total / 2 {
        tmp -= nums[ans].1;
        ans += 1;
    }
    ans as i32
}

pub fn min_steps(s: String, t: String) -> i32 {
    let mut scnt = [0i32; 26];
    let mut tcnt = [0i32; 26];
    s.chars().for_each(|c| scnt[c as usize - 'a' as usize] += 1);
    t.chars().for_each(|c| tcnt[c as usize - 'a' as usize] += 1);
    scnt.iter()
        .zip(tcnt.iter())
        .fold(0, |acc, (&s, &t)| acc + (s - t).abs())
}

pub fn truncate_sentence(s: String, k: i32) -> String {
    let s = s.split(" ").collect::<Vec<_>>();
    let mut ans = String::new();
    for i in 0..k {
        if i != 0 {
            ans = ans + " ";
        }
        ans = ans + s[i as usize];
    }
    ans
}

pub fn evaluate(s: String, knowledge: Vec<Vec<String>>) -> String {
    let mut hp = HashMap::new();
    knowledge.iter().for_each(|s| {
        hp.insert(s[0].clone(), s[1].clone());
    });
    let mut ans = String::new();
    let s1 = s.split("(").collect::<Vec<_>>();
    ans += s1[0];
    for i in 1..s1.len() {
        let s2 = s1[i].split(")").collect::<Vec<_>>();
        ans = ans
            + match hp.get(&s2[0].to_owned()) {
                Some(ss) => ss,
                None => "?",
            };
        ans += s2[1];
    }
    ans
}

pub fn reformat(s: String) -> String {
    let mut s1 = s
        .clone()
        .chars()
        .filter(|c| c.is_ascii_lowercase())
        .collect::<Vec<_>>();
    let mut s2 = s
        .clone()
        .chars()
        .filter(|c| c.is_ascii_digit())
        .collect::<Vec<_>>();
    if (s1.len() as i32 - s2.len() as i32).abs() > 1 {
        return "".to_owned();
    }
    if s1.len() < s2.len() {
        let t = s1;
        s1 = s2;
        s2 = t;
    }
    let mut ans = String::new();
    let mut idx1 = 0;
    let mut idx2 = 0;
    while idx1 < s1.len() && idx2 < s2.len() {
        ans += &s1[idx1].to_string();
        ans += &s2[idx2].to_string();
        idx1 += 1;
        idx2 += 1;
    }
    if idx1 == s1.len() - 1 {
        ans += &s1[idx1].to_string();
    }
    ans
}

pub fn group_the_people(group_sizes: Vec<i32>) -> Vec<Vec<i32>> {
    let mut hp = HashMap::new();
    group_sizes.iter().enumerate().for_each(|(i, size)| {
        let e = hp.entry(*size).or_insert(vec![]);
        e.push(i as i32);
    });
    let mut ans = vec![];
    for (k, v) in hp {
        let mut tmp = vec![];
        let mut idx = 0;
        while idx < v.len() {
            tmp.push(v[idx]);
            idx += 1;
            if idx % k as usize == 0 {
                ans.push(tmp.clone());
                tmp.drain(..);
            }
        }
    }
    ans
}

pub fn minimum_replacement(nums: Vec<i32>) -> i64 {
    let mut max = nums[nums.len() - 1];
    let mut ans = 0;
    for i in (0..nums.len() - 1).rev() {
        if nums[i] <= max {
            max = nums[i];
        } else {
            let x = (nums[i] + max - 1) / max;
            ans += x as i64 - 1;
            max = (nums[i]) / x;
        }
    }
    ans
}

pub fn largest_divisible_subset(nums: Vec<i32>) -> Vec<i32> {
    let mut ans = vec![];
    let mut nums = nums;
    nums.sort();
    let mut dp = vec![1; nums.len()];
    for i in 0..nums.len() {
        for j in 0..i {
            if nums[i] % nums[j] == 0 {
                dp[i] = dp[i].max(dp[j] + 1);
            }
        }
    }
    let mut max = *dp.iter().max().unwrap();
    for i in (0..dp.len()).rev() {
        if dp[i] == max {
            if ans.is_empty() {
                ans.push(nums[i]);
                max -= 1;
            } else if ans.last().unwrap() % nums[i] == 0 {
                ans.push(nums[i]);
                max -= 1;
            }
        }
        if max == 0 {
            break;
        }
    }
    ans
}

pub fn length_of_lis(nums: Vec<i32>) -> i32 {
    let mut dp = vec![1; nums.len()];
    let nums = nums;
    for i in 0..nums.len() {
        for j in 0..i {
            if nums[j] < nums[i] {
                dp[i] = dp[i].max(dp[j] + 1);
            }
        }
    }
    dp.iter().max().unwrap().to_owned()
}

pub fn find_radius(houses: Vec<i32>, heaters: Vec<i32>) -> i32 {
    let mut heaters = heaters;
    heaters.sort();
    let mut r = 0;
    for h in houses {
        let idx = match heaters.binary_search(&h) {
            Ok(_) => continue,
            Err(i) => i,
        };
        if idx == 0 {
            r = r.max(heaters[idx] - h);
        } else if idx == heaters.len() {
            r = r.max(h - heaters[idx - 1]);
        } else {
            r = r.max((heaters[idx] - h).min(h - heaters[idx - 1]));
        }
    }
    r
}

pub fn total_hamming_distance(nums: Vec<i32>) -> i32 {
    let mut cnt = [0; 32];
    let l = nums.len() as i32;
    for n in nums {
        let mut n = n;
        let mut idx = 0;
        while n != 0 {
            cnt[idx] += n % 2;
            n /= 2;
            idx += 1;
        }
    }
    cnt.iter().map(|&n| n * (l - n)).sum::<i32>()
}

pub fn find_max_consecutive_ones(nums: Vec<i32>) -> i32 {
    nums.iter()
        .fold((0, 0), |(acc, max), &n| {
            if n == 1 {
                (acc + 1, max.max(acc + 1))
            } else {
                (0, max)
            }
        })
        .1
}

pub fn find_poisoned_duration(time_series: Vec<i32>, duration: i32) -> i32 {
    let mut ans = 0;
    for i in 0..time_series.len() {
        if i == 0 {
            ans += duration;
        } else {
            if time_series[i] <= time_series[i - 1] + duration - 1 {
                ans += time_series[i] - time_series[i - 1];
            } else {
                ans += duration;
            }
        }
    }
    ans
}

pub fn find_words(words: Vec<String>) -> Vec<String> {
    let mut ans = vec![];
    fn is_same_line(word: &String) -> bool {
        let mut pre = -1;
        let word = word.to_lowercase().chars().collect::<Vec<_>>();
        for c in word {
            match c {
                'q' | 'w' | 'e' | 'r' | 't' | 'y' | 'u' | 'i' | 'o' | 'p' => {
                    if pre == -1 {
                        pre = 1;
                    } else if pre != 1 {
                        return false;
                    }
                }
                'a' | 's' | 'd' | 'f' | 'g' | 'h' | 'j' | 'k' | 'l' => {
                    if pre == -1 {
                        pre = 2;
                    } else if pre != 2 {
                        return false;
                    }
                }
                'z' | 'x' | 'c' | 'v' | 'b' | 'n' | 'm' => {
                    if pre == -1 {
                        pre = 3;
                    } else if pre != 3 {
                        return false;
                    }
                }
                _ => {}
            }
        }
        true
    }
    for word in words {
        if is_same_line(&word) {
            ans.push(word);
        }
    }
    ans
}

pub fn max_score(s: String) -> i32 {
    let s = s.chars().collect::<Vec<_>>();
    let mut cnt1 = 0;
    let n = s.len();
    let mut acc0 = vec![0; n];
    let mut acc1 = vec![0; n];
    for i in 0..s.len() {
        if s[i] == '0' {
            if i == 0 {
                acc0[i] = 1;
                acc1[i] = 0;
            } else {
                acc0[i] = acc0[i - 1] + 1;
                acc1[i] = acc1[i - 1];
            }
        } else {
            cnt1 += 1;
            if i == 0 {
                acc0[i] = 0;
                acc1[i] = 1;
            } else {
                acc0[i] = acc0[i - 1];
                acc1[i] = acc1[i - 1] + 1;
            }
        }
    }
    let mut ans = acc0[0] + cnt1 - acc1[0];
    for i in 1..n - 1 {
        ans = ans.max(acc0[i] + cnt1 - acc1[i]);
    }
    ans
}

pub fn count_vowels(word: String) -> i64 {
    let n = word.len();
    let word = word.chars().collect::<Vec<_>>();
    fn is_vowel(c: char) -> bool {
        match c {
            'a' | 'e' | 'i' | 'o' | 'u' => true,
            _ => false,
        }
    }
    let mut ans = 0i64;
    for i in 0..n {
        if is_vowel(word[i]) {
            ans += (i as i64 - 1) * (n as i64 - i as i64);
        }
    }
    ans
}

pub fn max_equal_freq(nums: Vec<i32>) -> i32 {
    let mut hp_cnt = HashMap::new();
    let mut hp_freq: HashMap<i32, i32> = HashMap::new();
    let mut maxfreq = 0;
    let mut ans = 1;
    for i in 0..nums.len() {
        match hp_cnt.get(&nums[i]) {
            Some(n) => {
                let v = hp_freq.get_mut(&n).unwrap();
                *v -= 1;
                let e = hp_cnt.entry(nums[i]).or_insert(0);
                *e += 1;
                maxfreq = maxfreq.max(*e);
                let e = hp_freq
                    .entry(hp_cnt.get(&nums[i]).unwrap().to_owned())
                    .or_insert(0);
                *e += 1;
            }
            None => {
                let e = hp_cnt.entry(nums[i]).or_insert(0);
                *e += 1;
                maxfreq = maxfreq.max(*e);
                let e = hp_freq.entry(1).or_insert(0);
                *e += 1;
            }
        }
        if (maxfreq == 1)
            || (*hp_freq.get(&maxfreq).unwrap() == 1
                && hp_freq.get(&maxfreq).unwrap() * maxfreq
                    + hp_freq.get(&(maxfreq - 1)).unwrap() * (maxfreq - 1)
                    == i as i32 + 1)
            || (*hp_freq.get(&maxfreq).unwrap() * maxfreq == i as i32
                && *hp_freq.get(&1).unwrap() == 1)
        {
            ans = ans.max(i as i32 + 1);
        }
    }
    ans
}

pub fn busy_student(start_time: Vec<i32>, end_time: Vec<i32>, query_time: i32) -> i32 {
    let mut ans = 0;
    start_time.iter().zip(end_time.iter()).for_each(|(&s, &e)| {
        if query_time >= s && query_time <= e {
            ans += 1;
        }
    });
    ans
}

pub fn min_falling_path_sum(grid: Vec<Vec<i32>>) -> i32 {
    let m = grid.len();
    let n = grid[0].len();
    let mut dp = vec![vec![i32::MIN; n]; m];
    dp[0].copy_from_slice(&grid[0]);
    let mut pre_min = i32::MAX;
    let mut pre_idx = -1;
    for i in 0..n {
        if dp[0][i] < pre_min {
            pre_min = dp[0][i];
            pre_idx = i as i32;
        }
    }
    for i in 1..m {
        let mut cur_min = i32::MAX;
        let mut cur_idx = i32::MAX;
        for j in 0..n {
            if j as i32 != pre_idx {
                dp[i][j] = grid[i][j] + pre_min;
                if dp[i][j] < cur_min {
                    cur_min = dp[i][j];
                    cur_idx = j as i32;
                }
            } else {
                let mut next_min = i32::MAX;
                for k in 0..n {
                    if k != j && dp[i - 1][k] < next_min {
                        next_min = dp[i - 1][k];
                    }
                }
                dp[i][j] = grid[i][j] + next_min;
                if dp[i][j] < cur_min {
                    cur_min = dp[i][j];
                    cur_idx = j as i32;
                }
            }
        }
        pre_min = cur_min;
        pre_idx = cur_idx;
    }
    *dp[m - 1].iter().min().unwrap()
}

pub fn max_product_path(grid: Vec<Vec<i32>>) -> i32 {
    let (m, n) = (grid.len(), grid[0].len());
    let mut maxes = vec![vec![0i64; n]; m];
    let mut mines = vec![vec![0i64; n]; m];
    for i in 0..n {
        if i == 0 {
            maxes[0][i] = grid[0][0] as i64;
            mines[0][i] = grid[0][0] as i64;
        } else {
            maxes[0][i] = grid[0][i] as i64 * maxes[0][i - 1];
            mines[0][i] = grid[0][i] as i64 * mines[0][i - 1];
        }
    }
    for i in 1..m {
        for j in 0..n {
            if grid[i][j] > 0 {
                if j == 0 {
                    maxes[i][j] = grid[i][j] as i64 * maxes[i - 1][j];
                    mines[i][j] = grid[i][j] as i64 * mines[i - 1][j];
                } else {
                    maxes[i][j] = grid[i][j] as i64 * maxes[i - 1][j].max(maxes[i][j - 1]);
                    mines[i][j] = grid[i][j] as i64 * mines[i - 1][j].min(mines[i][j - 1]);
                }
            } else {
                if j == 0 {
                    maxes[i][j] = grid[i][j] as i64 * mines[i - 1][j];
                    mines[i][j] = grid[i][j] as i64 * maxes[i - 1][j];
                } else {
                    maxes[i][j] = grid[i][j] as i64 * mines[i - 1][j].min(mines[i][j - 1]);
                    mines[i][j] = grid[i][j] as i64 * maxes[i - 1][j].max(maxes[i][j - 1]);
                }
            }
        }
    }
    if maxes[m - 1][n - 1] < 0 {
        return -1;
    } else {
        return (maxes[m - 1][n - 1] % 1_000_000_007) as i32;
    }
}

pub fn min_cut(s: String) -> i32 {
    let s = s.chars().collect::<Vec<_>>();
    let n = s.len();
    let mut g = vec![vec![true; n]; n];
    for i in (0..n).rev() {
        for j in i + 1..n {
            g[i][j] = g[i + 1][j - 1] && s[i] == s[j];
        }
    }
    let mut f = vec![i32::MAX; n];
    for i in 0..n {
        if g[0][i] {
            f[i] = 0;
        } else {
            for j in 0..i {
                if g[j + 1][i] {
                    f[i] = f[i].min(f[j] + 1);
                }
            }
        }
    }
    f[n - 1]
}

pub fn most_common_word(paragraph: String, banned: Vec<String>) -> String {
    let mut paragraph = paragraph.chars().collect::<Vec<_>>();
    let n = paragraph.len();
    for i in (0..n - 1).rev() {
        match paragraph[i] {
            '!' | '?' | '\'' | ',' | '.' => {
                if paragraph[i + 1] != ' ' {
                    paragraph.insert(i + 1, ' ');
                }
            }
            _ => {}
        }
    }
    let paragraph = paragraph
        .iter()
        .collect::<String>()
        .split(" ")
        .filter(|&s| s != "!" && s != "?" && s != ";" && s != "'" && s != "." && s != ",")
        .map(|s| {
            s.replace("?", "")
                .replace("!", "")
                .replace("'", "")
                .replace(",", "")
                .replace(";", "")
                .replace(".", "")
                .to_lowercase()
        })
        .collect::<Vec<_>>();
    let mut bp = BTreeMap::new();
    for p in paragraph {
        let e = bp.entry(p).or_insert(0);
        *e += 1;
    }
    let mut words = bp
        .into_iter()
        .filter(|w| !banned.contains(&w.0))
        .collect::<Vec<_>>();
    words.sort_by(|a, b| b.1.cmp(&a.1));
    words[0].0.clone()
}
