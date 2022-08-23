pub fn min_sum_square_diff(nums1: Vec<i32>, nums2: Vec<i32>, k1: i32, k2: i32) -> i64 {
    let mut diff = vec![];
    for i in 0..nums1.len() {
        diff.push((nums1[i] - nums2[i]).abs() as i64);
    }
    diff.push(0);
    let total = (k1 + k2) as i64;
    diff.sort();
    let mut helper = vec![0; diff.len()];
    for i in (0..diff.len()).rev() {
        if i == diff.len() - 1 {
            continue;
        } else if i > 0 {
            helper[i] = helper[i + 1] + (diff.len() - i - 1) as i64 * (diff[i + 1] - diff[i]);
        } else {
            helper[i] = helper[i + 1] + (diff.len() - 1) as i64 * (diff[i + 1]);
        }
    }
    // println!("diff: {:?}", diff);
    // println!("helper: {:?}", helper);
    if helper[0] <= total {
        return 0;
    }
    if total == 0 {
        let mut ans = 0;
        // println!("diff: {:?}", diff);
        for n in diff {
            ans += n * n;
        }
        return ans;
    }
    let mut l = 0;
    let mut r = diff.len() - 1;
    while l <= r {
        let mid = l + (r - l) / 2;
        // println!("l = {}, r = {}, mid = {}", l, r, mid);
        if helper[mid] < total {
            r = mid - 1;
        } else {
            l = mid + 1;
        }
    }
    // println!("r = {}", r);
    let remain;
    if r == diff.len() - 1 {
        remain = total;
    } else {
        remain = total - helper[r + 1];
    }
    let ave = remain / (diff.len() - 1 - r) as i64;
    let over = remain % (diff.len() - 1 - r) as i64;
    // println!("ave = {}, over = {}", ave, over);
    let base = diff[r + 1];
    for j in 0..over {
        diff[j as usize + r + 1] = (base - ave - 1).max(0);
    }
    for j in over..(diff.len() - r - 1) as i64 {
        diff[j as usize + r + 1] = (base - ave).max(0);
    }
    let mut ans = 0;
    // println!("diff: {:?}", diff);
    for n in diff {
        ans += n * n;
    }
    ans
}

#[test]
fn test_up() {
    use super::*;
    let nums1 = vec![18, 4, 8, 19, 13, 8];
    let nums2 = vec![18, 11, 8, 2, 13, 15];
    assert_eq!(17, min_sum_square_diff(nums1, nums2, 16, 8));
    let nums1 = vec![1, 2, 3, 4];
    let nums2 = vec![2, 10, 20, 19];
    assert_eq!(579, min_sum_square_diff(nums1, nums2, 0, 0));
    let nums1 = vec![11, 12, 13, 14, 15];
    let nums2 = vec![13, 16, 16, 12, 14];
    assert_eq!(3, min_sum_square_diff(nums1, nums2, 3, 6));
}

pub fn fill_cups(amount: Vec<i32>) -> i32 {
    let max = amount[0].max(amount[1].max(amount[2]));
    let av = (amount[0] + amount[1] + amount[2] + 1) / 2;
    max.max(av)
}

#[allow(dead_code)]
struct SmallestInfiniteSet {
    cnt: i32,
    record: Vec<i32>,
    small: i32,
}

#[allow(dead_code)]
impl SmallestInfiniteSet {
    fn new() -> Self {
        Self {
            cnt: 1000,
            record: vec![1; 1001],
            small: 1,
        }
    }

    fn pop_smallest(&mut self) -> i32 {
        self.record[self.small as usize] = 0;
        let ret = self.small;
        for i in self.small..=1000 {
            if self.record[i as usize] == 1 {
                self.small = i;
                break;
            }
        }
        ret
    }

    fn add_back(&mut self, num: i32) {
        if num < self.small {
            self.small = num;
            self.record[num as usize] = 1;
        } else {
            self.record[num as usize] = 1;
        }
    }
}

pub fn can_change(start: String, target: String) -> bool {
    let n = start.len();
    let start = start.chars().collect::<Vec<char>>();
    let target = target.chars().collect::<Vec<char>>();
    let mut idxs = 0;
    let mut idxt = 0;
    while idxt < n {
        while idxt < n && target[idxt] == '_' {
            idxt += 1;
        }
        if idxt == n {
            for j in idxs..n {
                if start[j] != '_' {
                    return false;
                }
            }
        } else {
            while idxs < n && start[idxs] == '_' {
                idxs += 1;
            }
            if idxs == n {
                return false;
            } else {
                if start[idxs] != target[idxt] {
                    return false;
                } else if start[idxs] == 'L' {
                    if idxs < idxt {
                        return false;
                    }
                } else if start[idxs] == 'R' {
                    if idxs > idxt {
                        return false;
                    }
                }
            }
        }
        idxt += 1;
        idxs += 1;
    }
    true
}

pub fn repeated_character(s: String) -> char {
    let mut cnt = vec![0; 26];
    let s = s.chars().map(|c| c as u8 - 'a' as u8).collect::<Vec<_>>();
    for i in 0..s.len() {
        cnt[s[i] as usize] += 1;
        if cnt[s[i] as usize] == 2 {
            return ('a' as u8 + s[i]) as char;
        }
    }
    'a' as char
}

pub fn equal_pairs(grid: Vec<Vec<i32>>) -> i32 {
    let mut ans = 0;
    let n = grid.len();
    for i in 0..n {
        for j in 0..n {
            let mut flag = true;
            for k in 0..n {
                if grid[i][k] != grid[k][j] {
                    flag = false;
                    break;
                }
            }
            if flag {
                ans += 1;
            }
        }
    }
    ans
}

use std::collections::{BTreeMap, HashMap, VecDeque};
#[allow(dead_code)]
struct FoodRatings {
    food2cui: BTreeMap<String, String>,
    food2rating: BTreeMap<String, i32>,
    cui2rating: HashMap<String, BTreeMap<String, i32>>,
    cui2max: HashMap<String, (String, i32)>,
}

#[allow(dead_code)]
impl FoodRatings {
    fn new(foods: Vec<String>, cuisines: Vec<String>, ratings: Vec<i32>) -> Self {
        let n = foods.len();
        let mut food2cui = BTreeMap::new();
        let mut food2rating = BTreeMap::new();
        let mut cui2rating = HashMap::new();
        let mut cui2max = HashMap::new();
        for i in 0..n {
            food2cui.insert(foods[i].clone(), cuisines[i].clone());
            food2rating.insert(foods[i].clone(), ratings[i]);
            let e = cui2rating
                .entry(cuisines[i].clone())
                .or_insert(BTreeMap::new());
            e.insert(foods[i].clone(), ratings[i]);
            let e = cui2max
                .entry(cuisines[i].clone())
                .or_insert(("".to_owned(), -1));
            if ratings[i] > e.1 {
                *e = (foods[i].clone(), ratings[i]);
            } else if ratings[i] == e.1 {
                match foods[i].cmp(&e.0) {
                    std::cmp::Ordering::Less => {
                        *e = (foods[i].clone(), ratings[i]);
                    }
                    _ => {}
                };
            }
        }
        Self {
            food2cui,
            food2rating,
            cui2rating,
            cui2max,
        }
    }

    fn change_rating(&mut self, food: String, new_rating: i32) {
        let cui = self.food2cui.get(&food).unwrap();
        self.food2rating.insert(food.clone(), new_rating);
        let e = self.cui2rating.get_mut(cui).unwrap();
        e.insert(food.clone(), new_rating);
        let e = self.cui2max.get_mut(&cui.clone()).unwrap();
        if new_rating > e.1 {
            *e = (food.clone(), new_rating);
        } else if new_rating == e.1 {
            match food.cmp(&e.0) {
                std::cmp::Ordering::Less => {
                    *e = (food.clone(), new_rating);
                }
                _ => {}
            }
        } else if food == e.0 {
            let e2 = self.cui2rating.get(cui).unwrap();
            let max = e2
                .iter()
                .max_by(|x, y| match x.1.cmp(y.1) {
                    std::cmp::Ordering::Equal => y.0.cmp(x.0),
                    _ => x.1.cmp(y.1),
                })
                .unwrap();
            *e = (max.0.clone(), *max.1);
        }
    }

    fn highest_rated(&self, cuisine: String) -> String {
        let max = self.cui2max.get(&cuisine).unwrap();
        max.0.clone()
    }
}

pub fn count_excellent_pairs(nums: Vec<i32>, k: i32) -> i64 {
    let mut ones = vec![];
    let mut vis = vec![];
    let n = nums.len();
    for i in 0..n {
        if !vis.contains(&nums[i]) {
            vis.push(nums[i]);
            let mut t = nums[i];
            let mut one = 0;
            while t != 0 {
                t = t & (t - 1);
                one += 1;
            }
            ones.push((nums[i], one));
        }
    }
    ones.sort_by_key(|x| x.1);
    let m = ones.len();
    let mut ans: i64 = 0;
    fn binary(ones: &Vec<(i32, i32)>, target: i32) -> i32 {
        let mut l = 0;
        let mut r = ones.len() as i32;
        while l < r {
            let mid = l + (r - l) / 2;
            if ones[mid as usize].1 < target {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        r
    }
    for i in 0..m {
        let tar = k - ones[i].1;
        let t = binary(&ones, tar);
        ans += m as i64 - t as i64;
    }
    ans
}

pub fn minimum_operations(nums: Vec<i32>) -> i32 {
    let mut map = vec![0; 101];
    for n in nums {
        if n != 0 {
            map[n as usize] = 1;
        }
    }
    let mut ans = 0;
    for n in map {
        if n >= 1 {
            ans += 1;
        }
    }
    ans
}

pub fn maximum_groups(grades: Vec<i32>) -> i32 {
    let n = grades.len() as i32;
    let mut ans = 0;
    let mut acc = 0;
    let mut round = 1;
    loop {
        acc += round;
        if n >= acc {
            ans += 1;
        } else {
            break;
        }
        round += 1;
    }
    ans
}

pub fn closest_meeting_node(edges: Vec<i32>, node1: i32, node2: i32) -> i32 {
    if node1 == node2 {
        return node1;
    }
    let mut hp1 = HashMap::new();
    let mut hp2 = HashMap::new();
    let mut vis1 = vec![];
    let mut vis2 = vec![];
    let mut tmp = node1;
    let mut dis = 0;
    hp1.insert(tmp, dis);
    loop {
        if edges[tmp as usize] == -1 || hp1.get(&edges[tmp as usize]).is_some() {
            break;
        }
        dis += 1;
        hp1.insert(edges[tmp as usize], dis);
        tmp = edges[tmp as usize];
    }
    tmp = node2;
    dis = 0;
    hp2.insert(tmp, dis);
    loop {
        if edges[tmp as usize] == -1 || hp2.get(&edges[tmp as usize]).is_some() {
            break;
        }
        dis += 1;
        hp2.insert(edges[tmp as usize], dis);
        tmp = edges[tmp as usize];
    }
    // println!("hp1 = {:?}, hp2 = {:?}", hp1, hp2);
    let mut p1 = node1;
    vis1.push(node1);
    let mut l = -1;
    let mut ans = -1;
    loop {
        if hp2.get(&p1).is_some() {
            let l1 = hp1.get(&p1).unwrap();
            let l2 = hp2.get(&p1).unwrap();
            l = (*l1).max(*l2);
            ans = p1;
            break;
        }
        if p1 == -1 {
            break;
        }
        p1 = edges[p1 as usize];
        if vis1.contains(&p1) {
            break;
        }
        vis1.push(p1);
    }
    if ans == -1 {
        return -1;
    }
    let mut p2 = node2;
    vis2.push(node2);
    loop {
        if hp1.get(&p2).is_some() {
            let l1 = hp1.get(&p2).unwrap();
            let l2 = hp2.get(&p2).unwrap();
            let ll = (*l1).max(*l2);
            if ll < l {
                ans = p2;
            } else if ll == l {
                ans = ans.min(p2);
            }
            break;
        }
        if p2 == -1 {
            break;
        }
        p2 = edges[p2 as usize];
        if vis2.contains(&p2) {
            break;
        }
        vis2.push(p2);
    }
    ans
}

#[test]
fn test_3() {
    use super::*;
    let edges = vec![2, 2, 3, -1];
    assert_eq!(2, closest_meeting_node(edges, 0, 1));
}

pub fn longest_cycle(edges: Vec<i32>) -> i32 {
    let n = edges.len();
    let mut vis = HashMap::new();
    let mut ans = -1;
    for i in 0..n {
        if vis.contains_key(&edges[i]) || edges[i] == -1 {
            continue;
        }
        let mut dic = HashMap::new();
        let mut l = 0;
        dic.insert(i, l);
        let mut k = i;
        while edges[k] != -1 {
            k = edges[k] as usize;
            l += 1;
            if dic.get(&k).is_some() {
                ans = ans.max(l - dic.get(&k).unwrap());
                break;
            }
            if vis.contains_key(&(k as i32)) {
                break;
            }
            dic.insert(k, l);
            vis.insert(k as i32, 1);
        }
    }
    ans
}

pub fn merge_similar_items(items1: Vec<Vec<i32>>, items2: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut hp = HashMap::new();
    for item in items1 {
        hp.insert(item[0], item[1]);
    }
    for item in items2 {
        let e = hp.entry(item[0]).or_insert(0);
        *e += item[1];
    }
    let mut ans = vec![];
    for (k, v) in hp {
        ans.push(vec![k, v]);
    }
    ans.sort_by_key(|x| x[0]);
    ans
}

pub fn count_bad_pairs(nums: Vec<i32>) -> i64 {
    let mut nums2 = vec![];
    let n = nums.len();
    for i in 0..n {
        nums2.push(nums[i] as i64 - i as i64);
    }
    let mut hp = HashMap::new();
    let n = nums.len() as i64;
    let mut cnt = n * (n - 1) / 2;
    for n in nums2 {
        let e = hp.get(&n);
        if e.is_none() {
            hp.insert(n, 1i64);
        } else {
            let e = e.unwrap().to_owned();
            cnt -= e;
            hp.insert(n, e + 1);
        }
    }
    cnt
}

pub fn task_scheduler_ii(tasks: Vec<i32>, space: i32) -> i64 {
    let mut day = 0i64;
    let mut hp = HashMap::new();
    let mut idx = 0;
    loop {
        if idx == tasks.len() {
            break;
        }
        if idx == 0 {
            day += 1;
            hp.insert(tasks[idx], day);
        } else {
            let e = hp.get(&tasks[idx]);
            if e.is_none() {
                day += 1;
                hp.insert(tasks[idx], day);
            } else {
                let e = e.unwrap().to_owned();
                if day - e <= space as i64 {
                    day = e + space as i64 + 1;
                    hp.insert(tasks[idx], day);
                } else {
                    day += 1;
                    hp.insert(tasks[idx], day);
                }
            }
        }
        idx += 1;
    }
    day
}

pub fn arithmetic_triplets(nums: Vec<i32>, diff: i32) -> i32 {
    let mut ans = 0;
    let n = nums.len();
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                if nums[j] - nums[i] == nums[k] - nums[j] && nums[j] - nums[i] == diff {
                    ans += 1;
                }
            }
        }
    }
    ans
}

pub fn reachable_nodes(n: i32, edges: Vec<Vec<i32>>, restricted: Vec<i32>) -> i32 {
    let mut vis = vec![0; n as usize];
    for i in 0..restricted.len() {
        vis[restricted[i] as usize] = 1;
    }
    let mut hp = HashMap::new();
    for edge in edges {
        let e = hp.entry(edge[0]).or_insert(vec![]);
        e.push(edge[1]);
        let e = hp.entry(edge[1]).or_insert(vec![]);
        e.push(edge[0]);
    }
    let mut q = VecDeque::new();
    let mut cnt = 1;
    q.push_back(0);
    let mut ans = 0;
    vis[0] = 1;
    while !q.is_empty() {
        let mut tmp = 0;
        for _ in 0..cnt {
            let node = q.pop_front().unwrap();
            ans += 1;
            let childs = hp.get(&node).unwrap();
            for child in childs {
                if vis[*child as usize] == 0 {
                    q.push_back(*child);
                    vis[*child as usize] = 1;
                    tmp += 1;
                }
            }
        }
        cnt = tmp;
    }
    ans
}

pub fn longest_ideal_string(s: String, k: i32) -> i32 {
    let mut cnt = vec![0; 26];
    let n = s.len();
    let s = s.chars().collect::<Vec<_>>();
    for i in 0..n {
        let idx = (s[i] as u8 - 'a' as u8) as i32;
        let l = 0.max(idx - k);
        let r = 25.min(idx + k);
        let mut max = cnt[l as usize];
        for i in l + 1..=r {
            if cnt[i as usize] > max {
                max = cnt[i as usize];
            }
        }
        cnt[idx as usize] = max + 1;
    }
    *cnt.iter().max().unwrap()
}

pub fn largest_local(grid: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let n = grid.len();
    let mut ans = vec![vec![0; n - 2]; n - 2];
    for i in 1..n - 1 {
        for j in 1..n - 1 {
            let mut max = grid[i - 1][j - 1];
            for k in 0..3 {
                for p in 0..3 {
                    max = max.max(grid[i - 1 + k][j - 1 + p]);
                }
            }
            ans[i - 1][j - 1] = max;
        }
    }
    ans
}

pub fn edge_score(edges: Vec<i32>) -> i32 {
    let mut cnt = vec![0i64; edges.len()];
    for i in 0..edges.len() {
        cnt[edges[i] as usize] += i as i64;
    }
    let mut maxcnt = i64::MIN;
    let mut ans = 0;
    for i in 0..cnt.len() {
        if cnt[i] > maxcnt {
            maxcnt = cnt[i];
            ans = i as i32;
        }
    }
    ans
}

#[test]
fn test_number() {
    use crate::competitions::smallest_number;
    let pattern = "IIIDIDDD".to_string();
    assert_eq!("123549876".to_string(), smallest_number(pattern));
}

pub fn smallest_number(pattern: String) -> String {
    let n = pattern.len();
    let pattern = pattern.chars().collect::<Vec<_>>();
    let mut ans = (1..=n + 1)
        .into_iter()
        .map(|n| n as i32)
        .collect::<Vec<i32>>();
    let mut i = 0;
    while i < n {
        if pattern[i] == 'I' {
            i += 1;
            continue;
        } else if pattern[i] == 'D' {
            let mut j = i;
            while j < n && pattern[j] == 'D' {
                j += 1;
            }
            ans[i..=j].reverse();
            i = j;
        }
    }
    ans.iter().map(|n| n.to_string()).collect()
}

#[test]
fn test_special() {
    use super::*;
    let n = 150;
    assert_eq!(123, count_special_numbers(n));
    assert_eq!(1005, count_special_numbers(1581));
}

pub fn count_special_numbers(n: i32) -> i32 {
    if n < 10 {
        return n;
    }
    let mut map = [9; 9];
    for i in 1..map.len() {
        map[i] = map[i - 1] * (9 - i as i32 + 1);
    }
    for i in 1..map.len() {
        map[i] += map[i - 1];
    }
    let mut ans = 0;
    let mut idx = 0;
    let mut n = n;
    let mut nums = vec![];
    while n > 0 {
        nums.push(n % 10);
        n /= 10;
        idx += 1;
    }
    nums.reverse();
    ans += map[idx - 2];
    for i in 0..idx {
        let mut tmp;
        if i == 0 {
            tmp = nums[0] - 1;
            for k in 0..idx as i32 - 1 {
                tmp = tmp * (9 - k);
            }
            ans += tmp;
        } else if i != idx - 1 {
            let mut cnt = 0;
            for k in 0..i {
                if nums[k] <= nums[i] - 1 {
                    cnt += 1;
                }
            }
            tmp = nums[i] - cnt;
            for k in 0..idx as i32 - i as i32 - 1 {
                tmp = tmp * (9 - k - (i as i32));
            }
            ans += tmp;
            let mut flag = false;
            for k in 0..i {
                if nums[k] == nums[i] {
                    flag = true;
                }
            }
            if flag {
                break;
            }
        } else {
            let mut cnt = 0;
            for k in 0..nums.len() - 1 {
                if nums[k] <= nums[nums.len() - 1] {
                    cnt += 1;
                }
            }
            tmp = nums[i] + 1 - cnt;
            ans += tmp;
        }
    }
    ans
}

pub fn minimum_recolors(blocks: String, k: i32) -> i32 {
    let blocks = blocks.chars().collect::<Vec<_>>();
    let mut ans = i32::MAX;
    for i in 0..=blocks.len() - k as usize {
        let mut cnt = 0;
        for j in i..(i + k as usize) {
            if blocks[j] == 'W' {
                cnt += 1;
            }
        }
        ans = ans.min(cnt);
    }
    ans
}

#[test]
fn test_08203() {
    use super::*;
    let s = "abc".to_string();
    let shifts = vec![vec![0, 1, 0], vec![1, 2, 1], vec![0, 2, 1]];
    assert_eq!("ace".to_string(), shifting_letters(s, shifts));
}

pub fn shifting_letters(s: String, shifts: Vec<Vec<i32>>) -> String {
    let mut s = s.chars().collect::<Vec<_>>();
    let mut map = vec![0; s.len() + 1];
    for i in 0..shifts.len() {
        if shifts[i][2] == 0 {
            map[shifts[i][0] as usize] -= 1;
            map[shifts[i][1] as usize + 1] += 1;
        } else {
            map[shifts[i][0] as usize] += 1;
            map[shifts[i][1] as usize + 1] -= 1;
        }
    }
    let mut pre: i32 = 0;
    for i in 0..s.len() {
        pre += map[i];
        let n = ((s[i] as u8 - 'a' as u8) as i32 + 26 * 10000 + pre) % 26;
        s[i] = ('a' as u8 + n as u8) as char;
    }
    s.iter().collect()
}

pub fn seconds_to_remove_occurrences(s: String) -> i32 {
    let mut ans = 0;
    let mut s = s;
    while let Some(_) = s.find("01") {
        ans += 1;
        let ss = s.replace("01", "10");
        s = ss;
    }
    ans
}

pub fn min_number_of_hours(
    initial_energy: i32,
    initial_experience: i32,
    energy: Vec<i32>,
    experience: Vec<i32>,
) -> i32 {
    let mut e = initial_energy;
    let mut exp = initial_experience;
    let mut exp_inc = 0;
    let mut e_inc = 0;
    for i in 0..energy.len() {
        e -= energy[i];
        if experience[i] >= exp {
            let inc = experience[i] - exp + 1;
            exp += inc;
            exp_inc += inc;
        }
        exp += experience[i];
    }
    if e <= 0 {
        e_inc = -e + 1;
    }
    e_inc + exp_inc
}

pub fn largest_palindromic(num: String) -> String {
    let nums = num.chars().collect::<Vec<_>>();
    let mut map = vec![0; 10];
    for i in 0..nums.len() {
        let idx = nums[i] as u8 - '0' as u8;
        map[idx as usize] += 1;
    }
    let mut flag0 = false;
    let mut flagodd = false;
    let mut max_idx = 0;
    let mut ans = vec![];
    let mut flag_exit = false;
    for i in (1..map.len()).rev() {
        if map[i] > 0 {
            flag_exit = true;
        }
        if map[i] > 1 {
            flag0 = true;
        }
        if map[i] % 2 == 1 {
            flagodd = true;
            max_idx = max_idx.max(i);
        }
        for _ in 0..map[i] / 2 {
            ans.push(('0' as u8 + i as u8) as char);
        }
    }
    if !flag_exit && map[0] != 0 {
        return "0".to_string();
    } else if !flag_exit && map[0] == 0 {
        return "".to_string();
    }
    if flag0 {
        if map[0] % 2 == 1 {
            flagodd = true;
            max_idx = max_idx.max(0);
        }
        for _ in 0..map[0] / 2 {
            ans.push('0');
        }
    }
    if flagodd {
        ans.push(('0' as u8 + max_idx as u8) as char);
    }
    if flag0 {
        if !flagodd {
            let n = ans.len();
            for i in (0..n).rev() {
                ans.push(ans[i]);
            }
        } else {
            let n = ans.len();
            for i in (0..n - 1).rev() {
                ans.push(ans[i]);
            }
        }
    }
    ans.iter().collect()
}
