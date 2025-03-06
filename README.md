<div id="top"></div>
<table>
  <tr>
    <td>
      <img alt="inferx" height="200px" src="res/ico.png">
    </td>
    <td>
      <h3>InferX</h3>
      <p>大语言模型异构推理引擎</p>
      <br>
      <a href="docs/"><strong>浏览文档  »</strong></a>
      <br>
      <br>
      <a href="docs/">查看 Demo</a>
      ·
      <a href="https://github.com/RuleEarth336123/InferX/issues">反馈 Bug</a>
      ·
      <a href="docs/">请求新功能</a>
    </td>
  </tr>
</table>



<!-- 目录 -->
<details>
  <summary>目录</summary>
  <ol>
    <li>
      <a href="#关于本项目">关于本项目</a>
      <ul>
        <li><a href="#构建工具">构建工具</a></li>
      </ul>
    </li>
    <li>
      <a href="#开始">开始</a>
      <ul>
        <li><a href="#依赖">依赖</a></li>
        <li><a href="#安装">安装</a></li>
      </ul>
    </li>
    <li><a href="#使用方法">使用方法</a></li>
    <li><a href="#路线图">路线图</a></li>
    <li><a href="#贡献">贡献</a></li>
    <li><a href="#许可证">许可证</a></li>
    <li><a href="#联系我们">联系我们</a></li>
    <li><a href="#致谢">致谢</a></li>
  </ol>
</details>



<!-- 关于本项目 -->
## 关于本项目

该框架是基于Transformer架构的大模型推理框架，提供了异构计算支持，能够在X86架构和NVIDIA GPU上加速推理过程。未来的改进方向可以使其具备跨平台编译的能力，从而满足大模型在端侧进行推理的需求。

<p align="right">(<a href="#top">返回顶部</a>)</p>

### 文件目录说明

```
llama-infer 
├── LICENSE.txt
├── README.md
├── /include/
│  ├── /base/
│  ├── /op/
│  ├── /tensor/
│  ├── /models/
│  ├── /samples/
│  ├── /helper/
├── /src/
│  ├── /base/
│  ├── ...
├── /kernels/
│  ├── /cpu/
│  ├── /cuda/
│  ├── kernel_interface
├── /docs/
│  ├── /rules/
│  │  ├── backend.txt
│  │  └── frontend.txt
├── /scripts/
├── /lib/
├── /thirds/
├── /log/
└── /util/

```
