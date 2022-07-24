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

struct SmallestInfiniteSet {
    cnt: i32,
    record: Vec<i32>,
    small: i32,
}

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

use std::collections::{BTreeMap, HashMap};
struct FoodRatings {
    food2cui: BTreeMap<String, String>,
    food2rating: BTreeMap<String, i32>,
    cui2rating: HashMap<String, BTreeMap<String, i32>>,
    cui2max: HashMap<String, (String, i32)>,
}

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
                .or_insert((BTreeMap::new()));
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
