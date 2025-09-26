# dataflow-doc

这个是[https://github.com/OpenDCAI/DataFlow](https://github.com/OpenDCAI/DataFlow)的文档。


## Install

```sh
npm i
```

## Usage

```sh
# start dev server 一般用这个，可以动态热渲染所有markdown的修改
npm run docs:dev
# build for production 这个主要是上传github之前测试下有无bug，有bug的话github无法渲染page的。
npm run docs:build
```

## 基本开发结构介绍：
基本都是一式两份，英语一份，汉语一份。

上方的导航栏配置主要在这个文件夹下：[navbars](./docs/.vuepress/navbars/)
各个文章的侧边栏主要是在这个文件夹下配置[sidebar](./docs/.vuepress/notes/)，可以手动可以自动，参考模板文档的[Sidebar配置](https://theme-plume.vuejs.press/config/theme/#sidebar)

如果开启了`npm run docs:dev`，在新建markdown的时候，会在markdown头部有一些配置前缀，这里简要介绍下：
```yaml
---
title: 框架设计 # 这个标题会用来作为sidebar的标题
createTime: 2025/06/13 14:59:56 # 不太重要
icon: material-symbols:deployed-code-outline # 可选，侧边栏展示时的小logo，从这个网址选烧包小logo https://icon-sets.iconify.design/
permalink: /zh/guide/framework/ # 这个自动生成的是8位码，可以自行修改以简明展示，注意不能和现有的其他md的路径重复。
---
```

## Documents

- [vuepress](https://vuepress.vuejs.org/)
- [vuepress-theme-plume](https://theme-plume.vuejs.press/)


## 好用插件：
- 重定向相关
npm i -D @vuepress/plugin-redirect@next
