import React from "react"
import { graphql } from "gatsby"
import Layout from "../components/layout"
import SEO from "../components/seo"
import CategoryBoard from "../components/category-board"
import RecipeBoard from "../components/recipe-board"

class IndexPage extends React.Component  {

  constructor(props) {
    super(props);
    // Specify state handle by the page
    //   - state is a map of selected categories
    const urlCategory = props.location.hash.substring(1)
    const selectionMap = props.data.allCategoriesYaml.edges.reduce((hmap, { node }) => {
        if (urlCategory !== "") {
            hmap[node.category] = urlCategory === node.category
        } else {
            hmap[node.category] = true
        }
        return hmap
      }, {})
    // Update state
    this.state = {
      selectedCategories: selectionMap
    }
    // Important: Bound context of this in the handler
    this.handleCategoryFilterChange = this.handleCategoryFilterChange.bind(this)
  }

  handleCategoryFilterChange = (category) => {
    this.setState((state, props) => {
      const newSelection = state.selectedCategories
      var allSelected = true
      for (const k in newSelection) allSelected &= newSelection[k]

      if (allSelected) {
          for (const k in newSelection) {
              if (k !== category) {
                  newSelection[k] = false
              }
          }
      } else {
        newSelection[category] = !newSelection[category]
      }
      return ({
        selectedCategories: newSelection
      })
    });
  }

  render() {
    const recipes = this.props.data.allRecipesYaml.edges.filter(({ node }) => this.state.selectedCategories[node.recipe.category])
    return (
      <Layout>
        <SEO title="Home" />
        <CategoryBoard categories={this.state.selectedCategories} toggleFunc={this.handleCategoryFilterChange} />
        <RecipeBoard recipes={recipes} />
      </Layout>
    )
  }
}

export const query =
    graphql`
      query {
        allCategoriesYaml {
          edges {
            node {
              category
            }
          }
        }
        allRecipesYaml {
          edges {
            node {
              recipe {
                category
                name
                url
                desc
              }
            }
          }
        }
      }
    `

export default IndexPage

//TODO: https://twitter.com/dan_abramov/status/824308413559668744?lang=en
//https://reactjs.org/docs/thinking-in-react.html
