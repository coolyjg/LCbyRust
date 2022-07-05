use crate::Solution;

// Definition for singly-linked list.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

#[allow(dead_code)]
impl ListNode {
    #[inline]
    fn new(val: i32) -> Self {
        ListNode { next: None, val }
    }
}

impl Solution {
    /// reverse a whole Linked-List
    pub fn reverse_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        let mut pre: Option<Box<ListNode>> = None;
        let mut cur: Option<Box<ListNode>> = head;
        while cur.is_some() {
            let mut node = cur.take().unwrap();
            cur = node.next;
            node.next = pre;
            pre = Some(node);
        }
        pre
    }

    /// reverse nodes between left and right(included) in a Linked-List
    pub fn reverse_between(
        head: Option<Box<ListNode>>,
        left: i32,
        right: i32,
    ) -> Option<Box<ListNode>> {
        let mut ret = Some(Box::new(ListNode {
            val: -1,
            next: head,
        }));
        let mut ret_mut = ret.as_mut();
        let mut rev = None;
        let mut cur = 0;
        while let Some(node) = ret_mut {
            cur += 1;
            while cur >= left && cur <= right {
                let mut next = node.next.take();
                node.next = next.as_mut().unwrap().next.take();
                next.as_mut().unwrap().next = rev.take();
                if cur < right {
                    rev = next;
                } else {
                    let mut rev_mut = next.as_mut();
                    while let Some(n) = rev_mut {
                        if n.next.is_none() {
                            n.next = node.next.take();
                            break;
                        }
                        rev_mut = n.next.as_mut();
                    }
                    node.next = next.take();
                    break;
                }
                cur += 1;
            }
            ret_mut = node.next.as_mut();
        }
        ret.as_mut().unwrap().next.take()
    }

    /// detect palindrome Linked-List
    pub fn is_palindrome(head: Option<Box<ListNode>>) -> bool {
        let mut nums = vec![];
        let mut cur = head.as_ref();
        while let Some(node) = cur {
            let val = node.val;
            nums.push(val);
            cur = node.next.as_ref();
        }
        let (mut l, mut r) = (0, nums.len() as i32 - 1);
        while l <= r {
            if nums[l as usize] != nums[r as usize] {
                return false;
            }
            l += 1;
            r -= 1;
        }
        true
    }

    /// delete specific node with value `val`
    pub fn delete_node(head: Option<Box<ListNode>>, val: i32) -> Option<Box<ListNode>> {
        let mut root = head;
        let mut cur = &mut root;
        while let Some(node) = cur {
            if node.val == val {
                *cur = node.next.take();
                break;
            }
            cur = &mut cur.as_mut().unwrap().next;
        }
        root
    }

    /// return values reversely
    pub fn reverse_print(head: Option<Box<ListNode>>) -> Vec<i32> {
        let mut ans = vec![];
        let mut cur = head;
        while let Some(node) = cur.as_mut() {
            ans.push(node.val);
            cur = node.next.take();
        }
        ans.into_iter().rev().collect::<Vec<i32>>()
    }

    /// get last kth node
    ///
    /// use cloned() to transport Option<&T> to Option<T>
    pub fn get_kth_from_end(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
        let mut start = head.as_ref();
        for _ in 0..k {
            start = start.unwrap().next.as_ref();
        }
        let mut tail = head.as_ref();
        while let Some(_) = start {
            start = start.unwrap().next.as_ref();
            tail = tail.unwrap().next.as_ref();
        }
        tail.cloned()
    }
}
