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

pub fn evaluate_tree(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    fn in_order(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        // if root.is_some(){
        let val = root.clone().unwrap().borrow().val;
        if val == 0 {
            return false;
        } else if val == 1 {
            return true;
        }
        let left = root.clone().unwrap().borrow().left.clone();
        let right = root.clone().unwrap().borrow().right.clone();
        let l = in_order(left);
        let r = in_order(right);
        if val == 2 {
            return l || r;
        } else {
            return l && r;
        }
        // }
        // false
    }
    in_order(root)
}

pub fn check_sub_tree(
    t1: Option<Rc<RefCell<TreeNode>>>,
    t2: Option<Rc<RefCell<TreeNode>>>,
) -> bool {
    if t1.is_none() && t2.is_none() {
        return true;
    }
    if t1.is_some() && t2.is_none() {
        return false;
    }
    if t1.is_none() && t2.is_some() {
        return false;
    }
    if t1.clone().unwrap().borrow().val != t2.clone().unwrap().borrow().val {
        return check_sub_tree(t1.clone().unwrap().borrow().left.clone(), t2.clone())
            || check_sub_tree(t1.clone().unwrap().borrow().right.clone(), t2.clone());
    } else {
        return (check_sub_tree(
            t1.clone().unwrap().borrow().left.clone(),
            t2.clone().unwrap().borrow().left.clone(),
        ) && check_sub_tree(
            t1.clone().unwrap().borrow().right.clone(),
            t2.clone().unwrap().borrow().right.clone(),
        )) || check_sub_tree(t1.clone().unwrap().borrow().left.clone(), t2.clone())
            || check_sub_tree(t1.clone().unwrap().borrow().right.clone(), t2.clone());
    }
}

pub fn prune_tree(root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
    let mut root = root;
    fn post_order(root: &mut Option<Rc<RefCell<TreeNode>>>) -> bool {
        if root.is_some() {
            let mut left = root.clone().unwrap().borrow().left.clone();
            let mut right = root.clone().unwrap().borrow().right.clone();
            let val = root.clone().unwrap().borrow().val;
            if !post_order(&mut left) {
                root.as_mut().unwrap().borrow_mut().left = None;
            }
            if !post_order(&mut right) {
                root.as_mut().unwrap().borrow_mut().right = None;
            }
            if !post_order(&mut left) && !post_order(&mut right) && val == 0 {
                return false;
            } else {
                return true;
            }
        }
        false
    }
    if post_order(&mut root) {
        return root;
    } else {
        let val = root.clone().unwrap().borrow().val;
        if val == 0 {
            return None;
        } else {
            return root;
        }
    }
}

#[allow(dead_code)]
struct CBTInserter {
    nodes: Vec<Option<Rc<RefCell<TreeNode>>>>,
}

#[allow(dead_code)]
impl CBTInserter {
    fn new(root: Option<Rc<RefCell<TreeNode>>>) -> Self {
        let mut vq = VecDeque::new();
        vq.push_back(root);
        let mut nodes = Vec::new();
        while !vq.is_empty() {
            let node = vq.pop_front().unwrap();
            let left = node.as_ref().unwrap().borrow().left.clone();
            let right = node.as_ref().unwrap().borrow().right.clone();
            if left.is_some() {
                vq.push_back(left);
            }
            if right.is_some() {
                vq.push_back(right);
            }
            nodes.push(node);
        }
        Self { nodes }
    }

    fn insert(&mut self, val: i32) -> i32 {
        let new_node = TreeNode::new(val);
        let n = self.nodes.len();
        self.nodes.push(Some(Rc::new(RefCell::new(new_node))));
        let father_idx = (n - 1) / 2;
        let mut father = self.nodes[father_idx].as_ref().unwrap().borrow_mut();
        if n % 2 == 1 {
            father.left = self.nodes[n].clone();
        } else {
            father.right = self.nodes[n].clone();
        }
        let val = father.val;
        return val;
    }

    fn get_root(&self) -> Option<Rc<RefCell<TreeNode>>> {
        self.nodes[0].clone()
    }
}

pub fn max_level_sum(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut sums = vec![];
    let mut q = VecDeque::new();
    q.push_back(root);
    let mut cnt = 1;
    while !q.is_empty() {
        let mut tmp = 0;
        let mut new_cnt = 0;
        for _ in 0..cnt {
            let node = q.pop_front().unwrap();
            let left = node.as_ref().unwrap().borrow().left.clone();
            let right = node.as_ref().unwrap().borrow().right.clone();
            let val = node.as_ref().unwrap().borrow().val;
            tmp += val;
            if left.is_some() {
                q.push_back(left);
                new_cnt += 1;
            }
            if right.is_some() {
                q.push_back(right);
                new_cnt += 1;
            }
        }
        sums.push(tmp);
        cnt = new_cnt;
    }
    let mut max = i32::MIN;
    let mut ans = 0;
    for (idx, sum) in sums.iter().enumerate() {
        if *sum > max {
            ans = idx + 1;
            max = *sum;
        }
    }
    ans as i32
}

pub fn add_one_row(
    root: Option<Rc<RefCell<TreeNode>>>,
    val: i32,
    depth: i32,
) -> Option<Rc<RefCell<TreeNode>>> {
    if depth == 1 {
        let mut ret = Some(Rc::new(RefCell::new(TreeNode::new(val))));
        ret.as_mut().unwrap().borrow_mut().left = root;
        return ret;
    }
    let mut levels = VecDeque::new();
    let mut cnt = 1;
    let root = root;
    let mut node_cnt = 1;
    levels.push_back(root.clone());
    while !levels.is_empty() {
        cnt += 1;
        if cnt == depth {
            while !levels.is_empty() {
                let node = levels.pop_front().unwrap();
                let left = node.clone().unwrap().borrow().left.clone();
                let right = node.clone().unwrap().borrow().right.clone();
                let mut new_node_left = Some(Rc::new(RefCell::new(TreeNode::new(val))));
                new_node_left.as_mut().unwrap().borrow_mut().left = left;
                let mut new_node_right = Some(Rc::new(RefCell::new(TreeNode::new(val))));
                new_node_right.as_mut().unwrap().borrow_mut().right = right;
                node.clone().unwrap().borrow_mut().left = new_node_left;
                node.clone().unwrap().borrow_mut().right = new_node_right;
            }
            break;
        } else {
            let mut tmp = 0;
            while node_cnt > 0 {
                node_cnt -= 1;
                let node = levels.pop_front().unwrap();
                let left = node.clone().unwrap().borrow().left.clone();
                let right = node.clone().unwrap().borrow().right.clone();
                if left.is_some() {
                    levels.push_back(left.clone());
                    tmp += 1;
                }
                if right.is_some() {
                    levels.push_back(right.clone());
                    tmp += 1;
                }
            }
            node_cnt = tmp;
        }
    }
    root
}

pub fn deepest_leaves_sum(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut q = VecDeque::new();
    let mut sum = 0;
    let mut cnt = 1;
    q.push_back(root);
    while !q.is_empty() {
        let mut n = 0;
        sum = 0;
        for _ in 0..cnt {
            let node = q.pop_front().unwrap();
            let val = node.clone().unwrap().borrow().val;
            let left = node.clone().unwrap().borrow().left.clone();
            let right = node.clone().unwrap().borrow().right.clone();
            if left.is_some() {
                q.push_back(left);
                n += 1;
            }
            if right.is_some() {
                q.push_back(right);
                n += 1;
            }
            sum += val;
        }
        cnt = n;
    }
    sum
}

pub fn construct_maximum_binary_tree(nums: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
    fn helper(nums: &Vec<i32>, left: i32, right: i32) -> Option<Rc<RefCell<TreeNode>>> {
        if left > right {
            return None;
        }
        let mut max = i32::MIN;
        let mut idx = 0;
        for i in left..=right {
            if nums[i as usize] > max {
                max = nums[i as usize];
                idx = i;
            }
        }
        let mut node = TreeNode::new(nums[idx as usize]);
        node.left = helper(nums, left, idx - 1);
        node.right = helper(nums, idx + 1, right);
        Some(Rc::new(RefCell::new(node)))
    }
    let root = helper(&nums, 0, nums.len() as i32 - 1);
    root
}

pub fn print_tree(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<String>> {
    fn get_depth(root: &Option<Rc<RefCell<TreeNode>>>) -> usize {
        if let Some(r) = root {
            1 + get_depth(&r.borrow().left).max(get_depth(&r.borrow().right))
        } else {
            0
        }
    }
    let n = get_depth(&root);
    let m = (1 << n) - 1;
    let mut queue = vec![];
    let mut grid = vec![vec!["".to_string(); m]; n];

    if let Some(r) = root {
        queue.push((r, 0, (m - 1) / 2));

        while queue.len() > 0 {
            let mut tmp = vec![];

            for i in 0..queue.len() {
                let (ref node, row, col) = queue[i];
                grid[row][col] = format!("{}", node.borrow().val);

                if let Some(left) = node.borrow_mut().left.take() {
                    tmp.push((left, row + 1, col - (1 << (n - row - 2))));
                }

                if let Some(right) = node.borrow_mut().right.take() {
                    tmp.push((right, row + 1, col + (1 << (n - row - 2))));
                }
            }

            queue = tmp;
        }
    }
    grid
}
