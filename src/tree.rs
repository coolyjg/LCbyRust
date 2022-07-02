use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<Rc<RefCell<TreeNode>>>,
    pub right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None,
        }
    }
}

pub fn good_nodes(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn count(root: Option<&Rc<RefCell<TreeNode>>>, max: i32) -> i32 {
        if root.is_none() {
            return 0;
        }
        let mut cnt = 0;
        let root = root.unwrap().clone();
        let v = { (*root).borrow().val };
        if v >= max {
            cnt += 1;
            cnt += {
                let root = root.clone();
                let root = (*root).borrow_mut();
                count(root.left.as_ref(), v)
            };
            cnt += {
                let root = root.clone();
                let root = (*root).borrow_mut();
                count(root.right.as_ref(), v)
            };
        } else {
            cnt += {
                let root = root.clone();
                let root = (*root).borrow_mut();
                count(root.left.as_ref(), max)
            };
            cnt += {
                let root = root.clone();
                let root = (*root).borrow_mut();
                count(root.right.as_ref(), max)
            };
        }
        cnt
    }
    let mut ans = 0;
    let root = root.unwrap().clone();
    let v = { (*root).borrow_mut().val };
    ans += count(Some(&root), v);
    ans
}

pub fn diameter_of_binary_tree(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    if root.is_none() {
        return 0;
    }
    let mut max_num = 0;
    dfs(root, &mut max_num);
    return max_num;
    fn dfs(root: Option<Rc<RefCell<TreeNode>>>, max_num: &mut i32) -> i32 {
        if root.is_none() {
            return 0;
        }
        let left_max = dfs(root.as_ref().unwrap().borrow().left.clone(), max_num);
        let right_max = dfs(root.as_ref().unwrap().borrow().right.clone(), max_num);
        if *max_num < left_max + right_max {
            *max_num = left_max + right_max;
        }
        return left_max.max(right_max) + 1;
    }
}

pub fn tree2str(root: Option<Rc<RefCell<TreeNode>>>) -> String {
    let mut ans = String::from("");
    fn helper(root: Option<Rc<RefCell<TreeNode>>>, ans: &mut String) {
        if root.is_some() {
            ans.push_str(&root.as_ref().unwrap().borrow().val.clone().to_string());
            ans.push('(');
            let len1 = ans.len();
            helper(root.as_ref().unwrap().borrow().left.clone(), ans);
            let mut left_is_none = false;
            if ans.len() == len1 {
                left_is_none = true;
            }
            ans.push_str(")(");
            let len2 = ans.len();
            helper(root.as_ref().unwrap().borrow().right.clone(), ans);
            if ans.len() == len2 && left_is_none {
                ans.pop();
                ans.pop();
                ans.pop();
            } else if ans.len() == len2 {
                ans.pop();
            } else {
                ans.push('(');
            }
        }
    }
    helper(root, &mut ans);
    ans
}

pub fn find_target(root: Option<Rc<RefCell<TreeNode>>>, k: i32) -> bool {
    let mut num = vec![];
    fn helper(root: Option<Rc<RefCell<TreeNode>>>, num: &mut Vec<i32>) {
        if root.is_some() {
            helper(root.as_ref().unwrap().borrow().left.clone(), num);
            num.push(root.as_ref().unwrap().borrow().val);
            helper(root.as_ref().unwrap().borrow().right.clone(), num);
        }
    }
    helper(root, &mut num);
    let (mut i, mut j) = (0, num.len() - 1);
    loop {
        if i >= j {
            break;
        }
        if num[i] + num[j] == k {
            return true;
        } else if num[i] + num[j] > k {
            j -= 1;
        } else if num[i] + num[j] < k {
            i += 1;
        }
    }
    false
}

pub fn is_unival_tree(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    let val = root.clone().unwrap().borrow().val;
    fn is_same(root: Option<Rc<RefCell<TreeNode>>>, val: i32) -> bool {
        if root.is_none() {
            return true;
        } else {
            if root.clone().unwrap().borrow().val != val {
                return false;
            }
            return is_same(root.clone().unwrap().borrow().left.clone(), val)
                && is_same(root.clone().unwrap().borrow().right.clone(), val);
        }
    }
    is_same(root.clone(), val)
}

pub fn sum_root_to_leaf(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn pre_order(tmp: i32, root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        let left = root.clone().unwrap().borrow().left.clone();
        let right = root.clone().unwrap().borrow().right.clone();
        let val = root.clone().unwrap().borrow().val;
        let mut ret = 0;
        if left.is_none() && right.is_none() {
            ret += tmp * 2 + val;
            return ret;
        }
        if left.is_some() {
            ret += pre_order(tmp * 2 + val, left);
        }
        if right.is_some() {
            ret += pre_order(tmp * 2 + val, right);
        }
        ret
    }
    pre_order(0, root)
}

pub fn find_frequent_tree_sum(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    fn dfs(root: Option<Rc<RefCell<TreeNode>>>, hp: &mut HashMap<i32, i32>) -> i32 {
        let mut sum = 0;
        let left = root.clone().unwrap().borrow().left.clone();
        let right = root.clone().unwrap().borrow().right.clone();
        let val = root.clone().unwrap().borrow().val;
        if left.is_some() {
            sum += dfs(left, hp);
        }
        if right.is_some() {
            sum += dfs(right, hp);
        }
        sum += val;
        let e = hp.entry(sum).or_insert(0);
        *e += 1;
        return sum;
    }
    let mut hp = HashMap::new();
    dfs(root, &mut hp);
    let mut ans = vec![];
    let mut max = i32::MAX;
    for (k, v) in hp.iter() {
        if max == i32::MAX {
            max = *v;
            ans.push(*k);
        } else {
            if *v == max {
                ans.push(*k);
            } else if *v > max {
                max = *v;
                ans.drain(..);
                ans.push(*k);
            }
        }
    }
    ans
}

pub fn find_bottom_left_value(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut q = vec![];
    q.push(root.clone());
    let mut cnt = 1;
    let mut ret = root.clone().unwrap().borrow().val;
    while !q.is_empty() {
        let mut t = 0;
        for i in 0..cnt {
            let node = q[i].clone();
            if i == 0 {
                ret = node.clone().unwrap().borrow().val;
            }
            let left = node.clone().unwrap().borrow().left.clone();
            let right = node.clone().unwrap().borrow().right.clone();
            if left.is_some() {
                q.push(left);
                t += 1;
            }
            if right.is_some() {
                q.push(right);
                t += 1;
            }
        }
        q.drain(0..cnt);
        cnt = t;
    }
    ret
}

pub fn largest_values(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    if root.is_none() {
        return vec![];
    }
    let mut ans = vec![];
    let mut dq = VecDeque::new();
    dq.push_back(root.clone());
    let mut idx = 0;
    let mut cnt = 1;
    while !dq.is_empty() {
        let mut temp = 0;
        for i in 0..cnt {
            let node = dq.pop_front().unwrap();
            if i == 0 {
                ans.push(node.clone().unwrap().borrow().val);
            } else {
                ans[idx] = ans[idx].max(node.clone().unwrap().borrow().val);
            }
            let (l, r) = (
                node.clone().unwrap().borrow().left.clone(),
                node.clone().unwrap().borrow().right.clone(),
            );
            if l.is_some() {
                dq.push_back(l);
                temp += 1;
            }
            if r.is_some() {
                dq.push_back(r);
                temp += 1;
            }
        }
        idx += 1;
        cnt = temp;
    }
    ans
}

pub fn sum_numbers(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn dfs(root: Option<Rc<RefCell<TreeNode>>>, tmp: i32, ans: &mut i32) {
        let l = root.clone().unwrap().borrow().left.clone();
        let r = root.clone().unwrap().borrow().right.clone();
        let val = root.clone().unwrap().borrow().val;
        if l.is_none() && r.is_none() {
            *ans += tmp * 10 + val;
        }
        if l.is_some() {
            dfs(l, tmp * 10 + val, ans);
        }
        if r.is_some() {
            dfs(r, tmp * 10 + val, ans);
        }
    }
    let mut ans = 0;
    dfs(root, 0, &mut ans);
    ans
}

pub fn postorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let mut ans = vec![];
    fn post_order(root: Option<Rc<RefCell<TreeNode>>>, ans: &mut Vec<i32>) {
        if root.is_some() {
            post_order(root.clone().unwrap().borrow().left.clone(), ans);
            post_order(root.clone().unwrap().borrow().right.clone(), ans);
            ans.push(root.clone().unwrap().borrow().val);
        }
    }
    post_order(root, &mut ans);
    ans
}

pub fn count_nodes(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    if root.is_none() {
        return 0;
    }
    let mut ans = 0;
    let mut q = VecDeque::new();
    let mut cnt = 1;
    ans += cnt;
    q.push_back(root.clone());
    while !q.is_empty() {
        let mut tmp = 0;
        for _ in 0..cnt {
            let node = q.pop_front().unwrap();
            let l = node.clone().unwrap().borrow().left.clone();
            let r = node.clone().unwrap().borrow().right.clone();
            if l.is_some() {
                q.push_back(l);
                tmp += 1;
            }
            if r.is_some() {
                q.push_back(r);
                tmp += 1;
            }
        }
        cnt = tmp;
        ans += cnt;
    }
    ans
}
